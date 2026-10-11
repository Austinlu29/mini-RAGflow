# PDF 与布局解析

现有 `pdf_parser(path)` 和 `LayoutParser().parse(elements)` 调用方式保持可用。

主入库流程通过 `document.langchain` 接入 LangChain：

```python
from document.langchain import LayoutPDFLoader, LayoutDocumentSplitter

documents = LayoutPDFLoader("data/test.pdf").load()
chunks = LayoutDocumentSplitter(chunk_size=300, chunk_overlap=50).split_documents(documents)
for chunk in chunks:
    print(chunk.metadata["page_number"], chunk.page_content)
```

`LayoutPDFLoader` 实现 `BaseLoader`，每个布局块返回一个 `Document`；
`metadata` 包含绝对路径 `source`、`page_number`、`block_type` 和四个坐标。
`LayoutDocumentSplitter` 实现 `BaseDocumentTransformer`，提供 `split_documents`、
`transform_documents` 和 `atransform_documents`，复用原有句界切分规则，保留标题和表格。

```python
from document.pdf_parser import pdf_parser
from document.layout_parser import LayoutParser

elements = pdf_parser("data/test.pdf")
blocks = LayoutParser().parse(elements)
```

`pdf_parser` 默认提取文本行及线框表格。表格元素携带 `table_id`、
`table_row`、`table_column`；ID 在每页重新编号，布局解析用页码和 ID
共同区分表格。空单元格保留，单元格内换行在表格块中展开为空格，
列用 ` | ` 分隔，行用换行分隔。表格内容不会重复输出为正文。
外置表头仍按普通文本处理，合并单元格的缺失位置保留为空值。

可选参数均为关键字参数：

```python
elements = pdf_parser(
    "report.pdf",
    password="reader-password",
    start_page=2,
    end_page=5,
)
```

页码从 1 开始，范围包含两端，输出保留原 PDF 页码。默认处理所有页。
文件读取、密码或页码错误会抛出异常；文档在处理异常时也会关闭。
表格检测遇到 `RuntimeError` 或 `ValueError` 时记录警告并保留文本提取。

默认标题识别根据正文字符数加权推断常用字号，再以其 1.6 倍、1.2 倍
分别判断 `Title` 和 `Heading`。不修改传入元素；保留已有类型标记。
只有单一字号的文档无法仅靠字号区分标题，可以指定绝对阈值：

```python
parser = LayoutParser(title_font_size=18, heading_font_size=13)
blocks = parser.parse(elements)
```

没有线框的表格仍可使用原来的坐标检测链路：

```python
from document.table_detector import TableDetector

elements = pdf_parser("report.pdf", detect_tables=False)
blocks = LayoutParser().parse(TableDetector().detect(elements))
```

`TableDetector` 也可接在默认 PDF 解析之后，会跳过已经识别的原生表格。
坐标推断依赖行列对齐和间距，复杂无边框表格可能误判；没有原生行列
信息时，无法可靠恢复空列或所有换行单元格。阅读排序按页、纵向、横向
坐标进行，复杂分栏、旋转文字和扫描 PDF 的 OCR 需要额外处理。

布局解析完成后，可直接切分为文档片段：

```python
from document.splitter import split_document

chunks = split_document(blocks, chunk_size=300, chunk_overlap=50)
for chunk in chunks:
    print(chunk.page_number, chunk.block_type, chunk.text)
```

长度和重叠量按字符数计算。正文优先在换行或句末切分，没有合适边界时
按长度截断；不同块不合并、不互相重叠。标题和表格保持完整，可能超过
指定长度。返回新的 `Block`，保留类型、页码和原始块坐标，不修改输入；
坐标仅用于追溯来源，不能代表片段的精确位置。空白块忽略。

默认参数为 `chunk_size=300`、`chunk_overlap=50`。在 `data/test.pdf` 上，
24 个正文块最长为 185 字，200、300、500、800 的切片内容完全相同，
均输出 38 个片段；100、150 分别拆开 8、2 个正文块。300 作为保留
完整段落并留有长度余量的基线，不代表已经验证过检索最优。表格最长
317 字，保持完整。当前实现不合并短段落，调整 size 无法消除孤立标题、
页脚等短片段。

验证命令（无需 pytest）：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

原生表格接口参见 [PyMuPDF Page 文档](https://pymupdf.readthedocs.io/en/latest/page.html#Page.find_tables)。
