import re
from typing import List

def extract_improvements(text: str) -> List[str]:
    improvements = re.findall(r'<improvement>(.*?)</improvement>', text)
    return improvements