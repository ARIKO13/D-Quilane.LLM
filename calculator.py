"""
Deterministic arithmetic for Qoolse AI — no dataset, no weights, no sampling.

A prompt is accepted only when every token is part of a math expression:
digits or Indonesian number words, the operators + - * / ^, parentheses, and
filler words such as "berapa" or "hasil dari". Anything else returns None, so
the caller falls through to retrieval or to the network instead of guessing.

    solve('lima tambah lima berapa')  -> 5 + 5 = 10
    solve('4 - 4 + 7 = ?')            -> 4 - 4 + 7 = 7
    solve('2 pangkat 10')             -> 2 ** 10 = 1024
    solve('jelaskan fotosintesis')    -> None

This is a tool call, not model output: production chat models get their
arithmetic right the same way, by routing the expression to a calculator.
"""
import ast
import re

MAX_EXPR_CHARS = 200
MAX_EXPONENT = 64
MAX_RESULT_BITS = 256

UNITS = {
    'nol': 0, 'satu': 1, 'dua': 2, 'tiga': 3, 'empat': 4, 'lima': 5,
    'enam': 6, 'tujuh': 7, 'delapan': 8, 'sembilan': 9,
}
SCALES = {
    'ribu': 1_000, 'seribu': 1_000, 'juta': 1_000_000,
    'miliar': 1_000_000_000, 'milyar': 1_000_000_000,
    'triliun': 1_000_000_000_000,
}
NUM_WORDS = set(UNITS) | set(SCALES) | {
    'sepuluh', 'sebelas', 'belas', 'puluh', 'ratus', 'seratus', 'seribu',
}
OPERATORS = {
    'tambah': '+', 'ditambah': '+', 'plus': '+', 'dijumlahkan': '+',
    'kurang': '-', 'dikurangi': '-', 'dikurang': '-', 'minus': '-',
    'kali': '*', 'dikali': '*', 'kaliya': '*', 'times': '*', 'x': '*',
    'bagi': '/', 'dibagi': '/', 'divided': '/',
    'pangkat': '**', 'kuadrat': '**2',
}
FILLER = {
    'berapa', 'berapakah', 'hasil', 'hasilnya', 'dari', 'daripada', 'adalah',
    'itu', 'hitung', 'hitunglah', 'tolong', 'coba', 'jadi', 'sama', 'dengan',
    'the', 'what', 'is', 'of', 'equals', 'equal', 'total', 'berapa?',
}
TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)?|[a-z]+|[()+\-*/^=?.]")
EXPR_RE = re.compile(r"^[\d\s+\-*/().]+$")
class TooLarge(ValueError):
    """Exponent outside the supported range."""


_BIN_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b,
            ast.Mult: lambda a, b: a * b, ast.Div: lambda a, b: a / b,
            ast.Pow: lambda a, b: a ** b}


def parse_number_words(words):
    """Parse an Indonesian numeral ('seratus dua puluh lima' -> 125).

    Returns None when the words do not form a well-formed numeral.
    """
    subtotal = 0
    pending = None
    total = 0
    for w in words:
        if w in UNITS:
            if pending is not None:
                return None
            pending = UNITS[w]
        elif w in ('sepuluh', 'sebelas'):
            if pending is not None:
                return None
            subtotal += 10 if w == 'sepuluh' else 11
        elif w == 'belas':
            if pending is None:
                return None
            subtotal += 10 + pending
            pending = None
        elif w in ('puluh', 'ratus'):
            unit = 10 if w == 'puluh' else 100
            subtotal += (pending if pending is not None else 1) * unit
            pending = None
        elif w == 'seratus':
            if pending is not None:
                return None
            subtotal += 100
        elif w in SCALES:
            chunk = subtotal + (pending or 0) or 1
            total += chunk * SCALES[w]
            subtotal = 0
            pending = None
        else:
            return None
    return total + subtotal + (pending or 0)


def _fmt(value):
    if isinstance(value, float):
        if value != value or value in (float('inf'), float('-inf')):
            return None
        if abs(value) < 1e15 and abs(value - round(value)) < 1e-12:
            return int(round(value))
        return round(value, 10)
    return value


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        val = _eval_node(node.operand)
        return -val if isinstance(node.op, ast.USub) else val
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise TooLarge('exponent too large')
        return _BIN_OPS[type(node.op)](left, right)
    raise ValueError(f'unsupported expression node: {type(node).__name__}')


def to_expression(prompt):
    """Return a canonical math expression string, or None if prompt is not math."""
    tokens = TOKEN_RE.findall(prompt.lower())[:400]
    pieces = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in NUM_WORDS:
            j = i
            run = []
            while j < len(tokens) and tokens[j] in NUM_WORDS:
                run.append(tokens[j])
                j += 1
            value = parse_number_words(run)
            if value is None:
                return None
            pieces.append(str(value))
            i = j
            continue
        if tok in OPERATORS:
            pieces.append(OPERATORS[tok])
        elif tok in '+-*/':
            pieces.append(tok)
        elif tok == '^':
            pieces.append('**')
        elif tok in FILLER or tok in '=?.':
            pass
        elif re.fullmatch(r"\d+(?:[.,]\d+)?", tok):
            pieces.append(tok.replace(',', '.'))
        elif tok in '()':
            pieces.append(tok)
        else:
            return None
        i += 1

    expr = ' '.join(pieces).strip()
    if not expr or len(expr) > MAX_EXPR_CHARS:
        return None
    if not re.search(r"\d", expr) or not re.search(r"[+\-*/]", expr):
        return None
    if not EXPR_RE.match(expr):
        return None
    return expr


def solve(prompt):
    """Return {'expr', 'value', 'text'} / {'expr', 'error'} / None."""
    expr = to_expression(prompt)
    if expr is None:
        return None
    try:
        tree = ast.parse(expr, mode='eval')
        value = _eval_node(tree)
    except ZeroDivisionError:
        return {'expr': expr, 'error': 'Pembagian dengan nol tidak terdefinisi.'}
    except TooLarge:
        return {'expr': expr,
                'error': f'Pangkat di atas {MAX_EXPONENT} tidak didukung kalkulator ini.'}
    except (SyntaxError, ValueError, OverflowError):
        return None
    if isinstance(value, int) and value.bit_length() > MAX_RESULT_BITS:
        return {'expr': expr, 'error': 'Hasilnya terlalu besar untuk dihitung.'}
    value = _fmt(value)
    if value is None:
        return {'expr': expr, 'error': 'Hasilnya bukan bilangan berhingga.'}
    canonical = ast.unparse(tree)
    return {'expr': canonical, 'value': value, 'text': f'{canonical} = {value}'}


if __name__ == '__main__':
    for sample in ['lima tambah lima berapa', '4 - 4 + 7 = ?', '12 kali 3',
                   'seratus dua puluh lima dibagi lima', '2 pangkat 10',
                   '100 / 0', 'jelaskan fotosintesis', '5 + 5 berapa?']:
        print(f'{sample!r:42} -> {solve(sample)}')
