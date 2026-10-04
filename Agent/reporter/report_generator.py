"""Versioned API output; legacy mixed-confidence tuples are refused."""
import csv
from pathlib import Path
from probability import probability


class ReportGenerator:
    fields = ('sample_id', 'title', 'predicted_label', 'p_fake', 'class_confidence',
              'actual_label', 'stage', 'status', 'error_type', 'explanation')

    def __init__(self, output_path):
        self.output_path = Path(output_path)

    def save_results(self, results, include_header=True):
        for row in results:
            if not isinstance(row, dict) or set(row) != set(self.fields):
                raise ValueError('Expected explicit probability/status fields; legacy tuples are unsupported')
            if row['status'] == 'success':
                p = probability(row['p_fake'])
                label = 'fake' if p >= .5 else 'real'
                if row['predicted_label'] != label or row['class_confidence'] != (p if label == 'fake' else 1-p):
                    raise ValueError('Inconsistent probability/label/confidence')
            elif (row['status'] != 'error' or row['p_fake'] is not None
                  or row['class_confidence'] is not None or row['predicted_label'] != 'abstain'):
                raise ValueError('Error rows must abstain with null probabilities')
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=self.fields)
            if include_header:
                writer.writeheader()
            writer.writerows(results)

    def print_accuracy(self, results):
        valid = [row for row in results if row['status'] == 'success']
        correct = sum(row['predicted_label'] == row['actual_label'] for row in valid)
        print(f'Samples={len(results)}, successful={len(valid)}, failed={len(results)-len(valid)}')
        if valid:
            print(f'Accuracy on successful samples: {correct / len(valid):.6f}')
