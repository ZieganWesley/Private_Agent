import zipfile

from private_agent.sandbox import SandboxManager
from private_agent.tools import calculator, documents


def _root(tmp_path):
    SandboxManager.set_root(str(tmp_path))
    return tmp_path


def test_create_docx(tmp_path):
    root = _root(tmp_path)
    out = root / "doc.docx"
    result = documents.create_docx.invoke(
        {"content": "Hello\nWorld", "save_path": str(out), "title": "T"}
    )
    assert "Created document" in result
    assert out.exists()
    with zipfile.ZipFile(out) as z:
        names = z.namelist()
    assert "word/document.xml" in names
    assert "[Content_Types].xml" in names


def test_create_xlsx(tmp_path):
    root = _root(tmp_path)
    out = root / "sheet.xlsx"
    result = documents.create_xlsx.invoke(
        {"rows": [["a", "b"], ["1", "2"]], "save_path": str(out)}
    )
    assert "Created spreadsheet" in result
    assert out.exists()
    with zipfile.ZipFile(out) as z:
        assert "xl/worksheets/sheet1.xml" in z.namelist()


def test_create_pdf(tmp_path):
    root = _root(tmp_path)
    out = root / "doc.pdf"
    result = documents.create_pdf.invoke(
        {"content": "Hello PDF", "save_path": str(out)}
    )
    assert "Created PDF" in result
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF-1.4")


def test_calculate_basic(tmp_path):
    _root(tmp_path)
    assert calculator.calculate.invoke({"expression": "2 + 3 * 4"}) == "14"
    assert calculator.calculate.invoke({"expression": "sqrt(16)"}) == "4.0"


def test_calculate_rejects_unsafe(tmp_path):
    _root(tmp_path)
    result = calculator.calculate.invoke(
        {"expression": "__import__('os').system('ls')"}
    )
    assert "Calculation error" in result


def test_search_documents_no_vectorstore():
    from private_agent.tools import rag_tools

    result = rag_tools.search_documents.invoke({"query": "hello"})
    assert "No document index" in result
