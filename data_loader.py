"""
Data Loader for SemEval-2014 ABSA Dataset
Parses XML format and creates train/test splits with implicit vs explicit classification
"""

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import argparse


@dataclass
class AspectOpinionPair:
    """Represents an aspect-opinion sentiment triplet"""
    aspect_term: str
    aspect_category: str
    opinion_term: Optional[str]
    polarity: str
    char_offset: Tuple[int, int]
    is_implicit: bool
    
    def to_dict(self):
        return {
            'aspect_term': self.aspect_term,
            'aspect_category': self.aspect_category,
            'opinion_term': self.opinion_term,
            'polarity': self.polarity,
            'char_offset': self.char_offset,
            'is_implicit': self.is_implicit
        }


@dataclass
class AspectSentiment:
    """Single aspect sentiment instance"""
    sentence_id: str
    sentence: str
    aspect_term: str
    aspect_category: str
    polarity: str
    is_implicit: bool
    opinion_term: Optional[str] = None
    
    def to_dict(self):
        return {
            'sentence_id': self.sentence_id,
            'sentence': self.sentence,
            'aspect_term': self.aspect_term,
            'aspect_category': self.aspect_category,
            'polarity': self.polarity,
            'is_implicit': self.is_implicit,
            'opinion_term': self.opinion_term
        }


class SemEvalParser:
    """Parse SemEval-2014 ABSA XML format"""
    
    @staticmethod
    def parse_xml(xml_path: str) -> List[Dict]:
        """
        Parse SemEval-2014 XML file
        
        Returns list of instances with format:
        {
            'sentence_id': str,
            'sentence': str,
            'aspects': [
                {
                    'aspect_term': str,
                    'aspect_category': str,
                    'polarity': str,
                    'opinion_term': Optional[str],
                    'char_offset': (start, end),
                    'is_implicit': bool
                }
            ]
        }
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        instances = []
        
        for sentence_elem in root.findall('.//sentence'):
            sentence_id = sentence_elem.get('id')
            text_elem = sentence_elem.find('text')
            if text_elem is None or text_elem.text is None:
                continue
            sentence_text = text_elem.text

            category_elements = sentence_elem.findall('./aspectCategories/aspectCategory')
            categories_by_polarity = defaultdict(list)
            for category in category_elements:
                category_name = category.get('category')
                category_polarity = category.get('polarity')
                if category_name and category_polarity:
                    categories_by_polarity[category_polarity.lower()].append(category_name)

            # Category-only annotations have no aspect term to use as a target.
            aspect_elements = sentence_elem.findall('./aspectTerms/aspectTerm')
            aspects = []
            for aspect in aspect_elements:
                aspect_term = aspect.get('term')
                polarity = aspect.get('polarity')
                if not aspect_term or not polarity:
                    continue
                polarity = polarity.lower()

                matching_categories = categories_by_polarity.get(polarity, [])
                aspect_category = matching_categories[0] if len(matching_categories) == 1 else None

                try:
                    char_from = int(aspect.get('from'))
                    char_to = int(aspect.get('to'))
                except (TypeError, ValueError):
                    char_from, char_to = -1, -1

                implicit_attr = aspect.get(
                    'implicit_sentiment', aspect.get('implicit', 'false')
                ).strip().lower()
                is_implicit = implicit_attr in {'true', 'y', '1'}
                opinion_term = aspect.get('opinion_words') or None

                aspects.append({
                    'aspect_term': aspect_term,
                    'aspect_category': aspect_category,
                    'polarity': polarity,
                    'opinion_term': opinion_term,
                    'char_offset': (char_from, char_to),
                    'is_implicit': is_implicit
                })
            
            instances.append({
                'sentence_id': sentence_id,
                'sentence': sentence_text,
                'aspects': aspects
            })
        
        return instances
    
    @staticmethod
    def flatten_to_instances(parsed_data: List[Dict]) -> List[AspectSentiment]:
        """
        Convert parsed sentences to individual aspect sentiment instances
        
        Each (sentence, aspect) pair becomes one training instance
        """
        instances = []
        
        for sentence_data in parsed_data:
            sentence_id = sentence_data['sentence_id']
            sentence = sentence_data['sentence']
            
            for aspect in sentence_data['aspects']:
                instance = AspectSentiment(
                    sentence_id=sentence_id,
                    sentence=sentence,
                    aspect_term=aspect['aspect_term'],
                    aspect_category=aspect['aspect_category'],
                    polarity=aspect['polarity'],
                    opinion_term=aspect['opinion_term'],
                    is_implicit=aspect['is_implicit']
                )
                instances.append(instance)
        
        return instances


class DatasetSplitter:
    """Split dataset by implicit/explicit"""
    
    @staticmethod
    def split_by_type(instances: List[AspectSentiment]) -> Tuple[List, List, List]:
        """
        Split instances into three groups:
        - Explicit: aspect_term != 'NULL'
        - Implicit: aspect_term == 'NULL'
        - All: both
        
        Returns: (all_instances, explicit_instances, implicit_instances)
        """
        explicit = [inst for inst in instances if not inst.is_implicit]
        implicit = [inst for inst in instances if inst.is_implicit]
        all_instances = instances
        
        return all_instances, explicit, implicit
    
    @staticmethod
    def statistics(instances: List[AspectSentiment]) -> Dict:
        """Get dataset statistics"""
        stats = {
            'total': len(instances),
            'implicit': len([i for i in instances if i.is_implicit]),
            'explicit': len([i for i in instances if not i.is_implicit]),
            'polarity_distribution': defaultdict(int),
            'aspect_categories': defaultdict(int)
        }
        
        for inst in instances:
            stats['polarity_distribution'][inst.polarity] += 1
            stats['aspect_categories'][inst.aspect_category] += 1
        
        return stats


class DataProcessor:
    """Process instances into training format"""
    
    @staticmethod
    def to_json(instances: List[AspectSentiment], output_path: str):
        """Save instances to JSON"""
        data = [inst.to_dict() for inst in instances]
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(data)} instances to {output_path}")
    
    @staticmethod
    def from_json(json_path: str) -> List[AspectSentiment]:
        """Load instances from JSON"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        instances = [
            AspectSentiment(
                sentence_id=inst['sentence_id'],
                sentence=inst['sentence'],
                aspect_term=inst['aspect_term'],
                aspect_category=inst['aspect_category'],
                polarity=inst['polarity'],
                is_implicit=inst['is_implicit'],
                opinion_term=inst.get('opinion_term')
            )
            for inst in data
        ]
        
        return instances
    
    @staticmethod
    def create_training_examples(
        instances: List[AspectSentiment],
        split_type: str = 'implicit'  # 'implicit', 'explicit', 'all'
    ) -> List[Dict]:
        """
        Convert instances to training examples
        
        Each example has:
        {
            'id': str,
            'sentence': str,
            'target': str (aspect_term),
            'gold_polarity': str,
            'aspect_category': str,
            'is_implicit': bool,
            'opinion_term': Optional[str]
        }
        """
        _, explicit, implicit = DatasetSplitter.split_by_type(instances)
        
        if split_type == 'implicit':
            data = implicit
        elif split_type == 'explicit':
            data = explicit
        else:  # 'all'
            data = instances
        
        examples = []
        for idx, inst in enumerate(data):
            example = {
                'id': f"{inst.sentence_id}_aspect_{idx}",
                'sentence': inst.sentence,
                'target': inst.aspect_term,  # Could be 'NULL' for implicit
                'gold_polarity': inst.polarity,
                'aspect_category': inst.aspect_category,
                'is_implicit': inst.is_implicit,
                'opinion_term': inst.opinion_term
            }
            examples.append(example)
        
        return examples


def main():
    parser = argparse.ArgumentParser(description='Process SemEval-2014 ABSA dataset')
    
    parser.add_argument('--train-path', type=str, default='data/semeval2014/Restaurants_Train.xml',
                        help='Path to training XML file')
    parser.add_argument('--test-path', type=str, default='data/semeval2014/Restaurants_Test.xml',
                        help='Path to test XML file')
    parser.add_argument('--output-dir', type=str, default='data/processed/',
                        help='Output directory for processed files')
    parser.add_argument('--output-prefix', type=str, default='',
                        help='Prefix for generated files, e.g. restaurant14_')
    parser.add_argument('--split-type', type=str, default='implicit',
                        choices=['implicit', 'explicit', 'all'],
                        help='Which instances to use')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("SemEval-2014 Dataset Processor")
    print("=" * 60)
    
    # Parse train
    print("\nParsing training data...")
    train_parsed = SemEvalParser.parse_xml(args.train_path)
    train_instances = SemEvalParser.flatten_to_instances(train_parsed)
    
    # Parse test
    print("Parsing test data...")
    test_parsed = SemEvalParser.parse_xml(args.test_path)
    test_instances = SemEvalParser.flatten_to_instances(test_parsed)
    
    # Get statistics
    print("\n" + "=" * 60)
    print("TRAINING SET STATISTICS")
    print("=" * 60)
    train_stats = DatasetSplitter.statistics(train_instances)
    print(f"Total instances: {train_stats['total']}")
    print(f"  - Implicit: {train_stats['implicit']}")
    print(f"  - Explicit: {train_stats['explicit']}")
    print(f"Polarity distribution: {dict(train_stats['polarity_distribution'])}")
    print(f"Aspect categories: {dict(train_stats['aspect_categories'])}")
    
    print("\n" + "=" * 60)
    print("TEST SET STATISTICS")
    print("=" * 60)
    test_stats = DatasetSplitter.statistics(test_instances)
    print(f"Total instances: {test_stats['total']}")
    print(f"  - Implicit: {test_stats['implicit']}")
    print(f"  - Explicit: {test_stats['explicit']}")
    print(f"Polarity distribution: {dict(test_stats['polarity_distribution'])}")
    print(f"Aspect categories: {dict(test_stats['aspect_categories'])}")
    
    # Create training examples
    print(f"\nCreating training examples (split_type={args.split_type})...")
    train_examples = DataProcessor.create_training_examples(train_instances, args.split_type)
    test_examples = DataProcessor.create_training_examples(test_instances, args.split_type)
    
    # Save
    train_output = output_dir / f'{args.output_prefix}train_{args.split_type}.json'
    test_output = output_dir / f'{args.output_prefix}test_{args.split_type}.json'
    
    DataProcessor.to_json(
        train_instances,
        str(output_dir / f'{args.output_prefix}train_full.json')
    )
    DataProcessor.to_json(
        test_instances,
        str(output_dir / f'{args.output_prefix}test_full.json')
    )
    
    # Save examples
    Path(train_output).parent.mkdir(parents=True, exist_ok=True)
    with open(train_output, 'w') as f:
        json.dump(train_examples, f, indent=2)
    print(f"Saved {len(train_examples)} training examples to {train_output}")
    
    with open(test_output, 'w') as f:
        json.dump(test_examples, f, indent=2)
    print(f"Saved {len(test_examples)} test examples to {test_output}")
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
