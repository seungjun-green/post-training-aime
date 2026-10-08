"""Small CPU regression gate bundled into the Colab; fails before GPU generation."""
from eval.final_answer import score_final


def main():
    cases = [
        (r'\boxed{42x^2-46x+12=0}', '21y^2-23y+6=0', 'Form a quadratic equation.', True),
        (r'\boxed{0}', '21y^2-23y+6=0', 'Form a quadratic equation.', False),
        (r'\boxed{x*(21x^2-23x+6)=0}', '21x^2-23x+6=0', 'Form a quadratic equation.', False),
        (r'\boxed{x^2+3x+2}', '(x+1)(x+2)', 'Factor completely.', False),
        (r'\boxed{(x+1)(x+2)}', '(x+1)(x+2)', 'Factor completely.', True),
        (r'\boxed{(x^2-1)(x+2)}', '(x-1)(x+1)(x+2)', 'Factor completely.', False),
        (r'\boxed{10010_2}', '(10010)_{2}', 'Subtract the binary numbers.', True),
        (r'\boxed{10011_2}', '(10010)_{2}', 'Subtract the binary numbers.', False),
        (r'\boxed{102_2}', '(10010)_{2}', 'Subtract the binary numbers.', False),
        (r'\boxed{))))}', '42', 'Compute a number.', False),
        (r'\boxed{这是答案}', '42', 'Compute a number.', False),
        ('Intermediate 42. Therefore the answer is 7.', '42', 'Compute a number.', False),
    ]
    for response, gold, problem, expected in cases:
        answer, correct, audit = score_final(response, gold, problem)
        if correct != expected:
            raise AssertionError(f'Grading regression: {response!r}, {audit}')
    print(f'Passed {len(cases)} CPU grading regression checks. No model generation performed.')


if __name__ == '__main__':
    main()
