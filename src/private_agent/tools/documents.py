"""Built-in document creation tools (Markdown/text to .docx, .xlsx and .pdf)."""

import zipfile
from xml.sax.saxutils import escape as xml_escape

from langchain_core.tools import tool

from ..sandbox import SandboxManager


def _validate_path(save_path: str):
    target = SandboxManager.validate_path(save_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


@tool
def create_docx(content: str, save_path: str, title: str = "") -> str:
    """Create a .docx file from plain text/Markdown-like content."""
    try:
        target = _validate_path(save_path)
        if not target.name.lower().endswith(".docx"):
            target = target.with_suffix(".docx")
        paragraphs = content.split("\n")
        body = ""
        if title:
            body += (
                "<w:p><w:r><w:rPr><w:b/></w:rPr>"
                f'<w:t xml:space="preserve">{xml_escape(title)}</w:t></w:r></w:p>'
            )
        for paragraph in paragraphs:
            body += (
                "<w:p><w:r>"
                f'<w:t xml:space="preserve">{xml_escape(paragraph)}</w:t>'
                "</w:r></w:p>"
            )
        document_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            f"<w:body>{body}<w:sectPr/></w:body></w:document>"
        )
        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/word/document.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
            "</Types>"
        )
        rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="word/document.xml"/>'
            "</Relationships>"
        )
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", content_types)
            archive.writestr("_rels/.rels", rels)
            archive.writestr("word/document.xml", document_xml)
        return f"Created document: {target}"
    except (OSError, ValueError) as exc:
        return f"Error creating docx: {exc}"


@tool
def create_xlsx(rows: list[list[str]] | None, save_path: str, title: str = "") -> str:
    """Create a simple .xlsx file from a list of rows (each row a list of cell values)."""
    try:
        target = _validate_path(save_path)
        if not target.name.lower().endswith(".xlsx"):
            target = target.with_suffix(".xlsx")
        if rows is None:
            rows = []
        sheet_rows = ""
        for row in rows:
            cells = ""
            for value in row:
                cells += (
                    '<c t="inlineStr"><is><t xml:space="preserve">'
                    f"{xml_escape(str(value))}</t></is></c>"
                )
            sheet_rows += f"<row>{cells}</row>"
        if title:
            sheet_rows = (
                '<row><c t="inlineStr"><is><t xml:space="preserve">'
                f"{xml_escape(title)}</t></is></c></row>" + sheet_rows
            )
        sheet_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            f"<sheetData>{sheet_rows}</sheetData></worksheet>"
        )
        workbook_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets></workbook>'
        )
        content_types = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            "</Types>"
        )
        rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
            'Target="xl/workbook.xml"/>'
            "</Relationships>"
        )
        wb_rels = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            'Target="worksheets/sheet1.xml"/>'
            "</Relationships>"
        )
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("[Content_Types].xml", content_types)
            archive.writestr("_rels/.rels", rels)
            archive.writestr("xl/workbook.xml", workbook_xml)
            archive.writestr("xl/_rels/workbook.xml.rels", wb_rels)
            archive.writestr("xl/worksheets/sheet1.xml", sheet_xml)
        return f"Created spreadsheet: {target}"
    except (OSError, ValueError) as exc:
        return f"Error creating xlsx: {exc}"


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


@tool
def create_pdf(content: str, save_path: str, title: str = "") -> str:
    """Create a simple text-based .pdf file."""
    try:
        target = _validate_path(save_path)
        if not target.name.lower().endswith(".pdf"):
            target = target.with_suffix(".pdf")
        lines = []
        if title:
            lines.append(title)
            lines.append("")
        lines.extend(content.split("\n"))
        text_ops = []
        y = 760
        for line in lines:
            text_ops.append(f"BT /F1 11 Tf 72 {y} Td ({_escape_pdf_text(line)}) Tj ET")
            y -= 14
            if y < 72:
                break
        stream = "\n".join(text_ops).encode("latin-1", errors="replace")
        objects = []
        objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
        objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
        objects.append(
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>\nendobj\n"
        )
        objects.append(
            b"4 0 obj\n<< /Length %d >>\nstream\n" % len(stream)
            + stream
            + b"\nendstream\nendobj\n"
        )
        objects.append(
            b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        )
        offsets = []
        # compute offsets
        # rebuild with offsets properly
        buf = b"%PDF-1.4\n"
        offsets = [0]
        for obj in objects:
            offsets.append(len(buf))
            buf += obj
        xref_pos = len(buf)
        buf += b"xref\n0 %d\n" % (len(objects) + 1)
        buf += b"0000000000 65535 f \n"
        for i in range(1, len(objects) + 1):
            buf += b"%010d 00000 n \n" % offsets[i]
        buf += b"trailer\n<< /Size %d /Root 1 0 R >>\n" % (len(objects) + 1)
        buf += b"startxref\n%d\n%%%%EOF\n" % xref_pos
        target.write_bytes(buf)
        return f"Created PDF: {target}"
    except (OSError, ValueError) as exc:
        return f"Error creating pdf: {exc}"
