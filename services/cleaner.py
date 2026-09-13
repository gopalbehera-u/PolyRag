import re 


def clean_text(text:str) -> str:
    if not text:
        return ""

    # Normalize Windows/Mac line endings
    text=text.replace("\r\n","\n").replace("\r","\n")

    # Tabs -> single space
    text=text.replace("\t"," ")

    #Collapse runs horizontal whitespace
    text=re.sub(r"[\u00A0]+"," ",text)

     # Collapse 3+ newlines down to a paragraph break (2 newlines)

    text=re.sub(r"\n{3,}","\n\n",text)


    # A single newline that is NOT part of a paragraph break is usually a
    # soft line-wrap from PDF extraction -> turn it into a space, but keep
    # genuine paragraph breaks (\n\n) intact.

    text=re.sub(r"(?<!\n)\n(?!\n)"," ",text)


    # Trim training space on each remaining line 
    text ="\n".join(line.strip()  for line in text.split("\n"))

    return text.strip()