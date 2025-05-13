from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_text_splitters import CharacterTextSplitter


def chunk_with_md_header(text, md_splitter: MarkdownHeaderTextSplitter):
    """
    Splits the text into chunks based on Markdown headers.
    """
    return md_splitter.split_text(text)


def chunk_with_character(text, char_splitter: CharacterTextSplitter):
    """
    Splits the text into chunks based on character length.
    """
    return char_splitter.split_text(text)


if __name__ == "__main__":
    with open("documents/usfs_fia_timberland_and_forestland.md", "r", encoding="utf-8") as f:
        text = f.read()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]

    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

    md_splitted = chunk_with_md_header(text, md_splitter)

    print(md_splitted[0].id)
