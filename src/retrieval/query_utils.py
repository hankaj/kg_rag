from langchain_neo4j.vectorstores.neo4j_vector import remove_lucene_chars


def generate_full_text_query(input_text: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspellings.

    Args:
        input_text: Input text to convert to a full-text query

    Returns:
        Full-text query string
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input_text).split() if el]

    if not words:
        return ""

    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"

    return full_text_query.strip()
