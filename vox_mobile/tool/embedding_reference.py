"""Reference token ids and vectors for EmbeddingGemma, made with Hugging
Face's own tokenizer and ONNX Runtime. CI checks the app's Dart pipeline
(test/integration/embedding_gemma_test.dart) against this file: a wrong
BOS/EOS or prompt convention changes every vector without any error.

usage: python3 embedding_reference.py <model dir> <out.json> [graph]
The model dir holds the graph (model_quantized.onnx by default; also
model_q4.onnx or model.onnx), its .onnx_data and tokenizer.json.
"""
import json
import sys

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

PROMPTS = {
    'query': 'task: search result | query: ',
    'document': 'title: none | text: ',
}

CASES = [
    ('query', 'money worries'),
    ('query', 'when is the dentist?'),
    ('query', 'Did Ericah say anything about the kids\' school?'),
    ('document', 'We cannot pay the electric bill this month'),
    ('document', 'Rent is due on Friday too'),
    ('document', 'The puppy chewed my shoe again'),
    ('document', 'The dentist appointment is on Tuesday at 9:30.'),
    ('document', 'When is the dentist?\nyes, Tuesday works'),
    ('document', 'Café au lait, naïve façade — déjà vu! 😂🎉'),
    ('document', 'I\'m gonna call Mom @ 555-0142 re: the $1,250 deposit (50%)'),
    ('document', '   leading and trailing spaces   '),
    ('document', 'Ünïcödé ß ø 日本語のテキスト 한국어'),
]


def main(model_dir: str, out: str, graph: str) -> None:
    tok = Tokenizer.from_file(f'{model_dir}/tokenizer.json')
    sess = ort.InferenceSession(f'{model_dir}/{graph}')
    cases = []
    for task, text in CASES:
        ids = tok.encode(PROMPTS[task] + text).ids  # with the file's own special tokens
        n = len(ids)
        vec = sess.run(['sentence_embedding'], {
            'input_ids': np.array([ids], dtype=np.int64),
            'attention_mask': np.ones((1, n), dtype=np.int64),
        })[0][0]
        vec = vec / np.linalg.norm(vec)
        cases.append({'task': task, 'text': text, 'ids': ids, 'vec': [float(x) for x in vec]})
        print(task, repr(text), 'ids', ids[:6], '...', ids[-3:], 'len', n)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'cases': cases}, f, ensure_ascii=False)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else 'model_quantized.onnx')
