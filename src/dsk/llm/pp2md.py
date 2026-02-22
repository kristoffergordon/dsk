"""Document converters for transforming PowerPoint presentations to markdown."""

import base64
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_mistralai import ChatMistralAI
from pdf2image import convert_from_path


@dataclass
class SlideResult:
    """Result of processing a single slide."""

    slide_number: int
    description: str


@dataclass
class PresentationResult:
    """Result of processing a complete presentation."""

    file_path: Path
    title: str
    markdown: str
    error: str | None = None


class PowerPointToMarkdownConverter:
    """Converts PowerPoint presentations (as PDFs) to markdown using vision AI.

    Supports parallel processing of multiple files and parallel slide analysis.
    """

    def __init__(
        self,
        slide_llm: ChatMistralAI | None = None,
        summary_llm: ChatMistralAI | None = None,
        dpi: int = 150,
        max_workers_files: int = 3,
        max_workers_slides: int = 5,
    ):
        """Initialize the converter.

        Args:
            slide_llm: LLM for slide descriptions (uses Pixtral by default)
            summary_llm: LLM for summary generation (uses Magistral by default)
            dpi: DPI for PDF to image conversion
            max_workers_files: Max parallel files to process
            max_workers_slides: Max parallel slides to process per file
        """
        self.slide_llm = slide_llm or ChatMistralAI(
            name="pixtral-12b-2409", temperature=0.2
        )
        self.summary_llm = summary_llm or ChatMistralAI(
            name="magistral-small-2509", temperature=0.2
        )
        self.dpi = dpi
        self.max_workers_files = max_workers_files
        self.max_workers_slides = max_workers_slides

    def _image_to_base64(self, img: Any) -> str:
        """Convert PIL Image to base64 string."""
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="PNG")
        img_bytes = img_byte_arr.getvalue()
        return base64.b64encode(img_bytes).decode("utf-8")

    def _describe_slide(self, img: Any, slide_number: int) -> SlideResult:
        """Describe a single slide using vision AI."""
        img_base64 = self._image_to_base64(img)

        message = HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": f"data:image/png;base64,{img_base64}",
                },
                {
                    "type": "text",
                    "text": "Beskriv denne PowerPoint-slide i detaljer. Inkluder alt tekstindhold, visuelle elementer, struktur og enhver vigtig information. Vær grundig og præcis.",
                },
            ]
        )

        response = self.slide_llm.invoke([message])
        # Ensure we get a string from the response
        description = str(response.content) if response.content else ""
        return SlideResult(slide_number=slide_number, description=description)

    def _generate_summary(self, slide_descriptions: list[str]) -> str:
        """Generate overall summary from slide descriptions."""
        summary_prompt = (
            "Baseret på følgende slide-for-slide beskrivelser af en PowerPoint-præsentation, "
            "giv et kortfattet 2-3 afsnit sammendrag, der fanger hovedformålet, nøgleemner, "
            "og det overordnede budskab i præsentationen:\n\n"
        )
        for i, desc in enumerate(slide_descriptions, 1):
            summary_prompt += f"**Slide {i}:** {desc}\n\n"

        summary_message = HumanMessage(content=summary_prompt)
        summary_response = self.summary_llm.invoke([summary_message])

        # Extract text from content blocks
        summary = ""
        for block in summary_response.content_blocks:
            if block["type"] == "text":
                summary = block["text"]
                break

        if not summary:
            raise ValueError("No summary generated")

        return summary

    def _format_markdown(
        self, title: str, summary: str, slide_descriptions: list[str]
    ) -> str:
        """Format results as markdown."""
        markdown = f"# {title}\n\n"
        markdown += f"## Summary\n\n{summary}\n\n"
        markdown += "---\n\n"

        for i, desc in enumerate(slide_descriptions, 1):
            markdown += f"## Slide {i}\n\n{desc}\n\n"

        return markdown

    def _process_single_file(self, pdf_path: Path) -> PresentationResult:
        """Process a single PowerPoint file (as PDF)."""
        try:
            print(f"\n[{pdf_path.name}] Converting PDF to images...")
            images = convert_from_path(pdf_path, dpi=self.dpi)
            print(f"[{pdf_path.name}] Generated {len(images)} slide images")

            title = pdf_path.stem

            # Process slides in parallel
            print(f"[{pdf_path.name}] Describing {len(images)} slides in parallel...")
            slide_results: list[SlideResult] = []

            with ThreadPoolExecutor(max_workers=self.max_workers_slides) as executor:
                future_to_slide = {
                    executor.submit(self._describe_slide, img, i): i
                    for i, img in enumerate(images, 1)
                }

                for future in as_completed(future_to_slide):
                    slide_num = future_to_slide[future]
                    try:
                        result = future.result()
                        slide_results.append(result)
                        print(
                            f"[{pdf_path.name}] Slide {slide_num}/{len(images)} "
                            f"described ({len(result.description)} chars)"
                        )
                    except Exception as e:
                        print(
                            f"[{pdf_path.name}] Error describing slide {slide_num}: {e}"
                        )
                        raise

            # Sort results by slide number
            slide_results.sort(key=lambda x: x.slide_number)
            slide_descriptions = [r.description for r in slide_results]

            # Generate summary
            print(f"[{pdf_path.name}] Generating summary...")
            summary = self._generate_summary(slide_descriptions)

            # Format as markdown
            markdown = self._format_markdown(title, summary, slide_descriptions)

            print(f"[{pdf_path.name}] ✓ Complete ({len(markdown)} chars)")

            return PresentationResult(
                file_path=pdf_path, title=title, markdown=markdown
            )

        except Exception as e:
            print(f"[{pdf_path.name}] ✗ Error: {e}")
            return PresentationResult(
                file_path=pdf_path,
                title=pdf_path.stem,
                markdown="",
                error=str(e),
            )

    def convert_files(
        self, pdf_paths: list[Path], save_dir: Path | None = None
    ) -> list[PresentationResult]:
        """Convert multiple PowerPoint files to markdown in parallel.

        Args:
            pdf_paths: List of paths to PDF files (exported PowerPoints)
            save_dir: Optional directory to save markdown files

        Returns:
            List of PresentationResult objects
        """
        print(f"Starting conversion of {len(pdf_paths)} files...")
        print(
            f"Parallel files: {self.max_workers_files}, Parallel slides: {self.max_workers_slides}"
        )

        results: list[PresentationResult] = []

        with ThreadPoolExecutor(max_workers=self.max_workers_files) as executor:
            future_to_file = {
                executor.submit(self._process_single_file, pdf_path): pdf_path
                for pdf_path in pdf_paths
            }

            for future in as_completed(future_to_file):
                result = future.result()
                results.append(result)

                # Save to file if save_dir is provided and no error occurred
                if save_dir and not result.error:
                    output_file = save_dir / f"{result.title}.md"
                    output_file.write_text(result.markdown)
                    print(f"[{result.file_path.name}] Saved to: {output_file}")

        # Print summary
        successful = sum(1 for r in results if not r.error)
        failed = len(results) - successful
        print(f"\n{'=' * 80}")
        print(f"Conversion complete: {successful} successful, {failed} failed")
        print(f"{'=' * 80}")

        return results
