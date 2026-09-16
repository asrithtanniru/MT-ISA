"""
Phase 1: Auxiliary Task Construction with Self-Refine
Implements the LLM-based generation with polarity intervention and confidence scoring
"""

import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
from tqdm import tqdm

from ollama_client import OllamaClient, AspectOpinionExtractor


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AuxiliaryData:
    """Auxiliary task data with confidence scores"""
    instance_id: str
    sentence: str
    target: str
    gold_polarity: str
    
    # Extracted by LLM
    aspect: str
    aspect_confidence: float
    opinion: str
    opinion_confidence: float
    
    # Metadata
    refinement_iterations: int
    converged: bool
    
    def to_dict(self):
        return asdict(self)


class AuxiliaryGenerator:
    """
    Generate auxiliary task data using self-refine with polarity intervention
    Algorithm 1 from MT-ISA paper
    """
    
    def __init__(
        self,
        ollama_client: OllamaClient,
        max_refinement_epochs: int = 10,
        confidence_threshold: float = 0.5
    ):
        """
        Args:
            ollama_client: Ollama client for LLM
            max_refinement_epochs: Maximum refinement iterations
            confidence_threshold: Minimum confidence to accept
        """
        self.extractor = AspectOpinionExtractor(ollama_client)
        self.max_refinement_epochs = max_refinement_epochs
        self.confidence_threshold = confidence_threshold
    
    def generate_for_instance(
        self,
        instance_id: str,
        sentence: str,
        target: str,
        gold_polarity: str,
        verbose: bool = False
    ) -> AuxiliaryData:
        """
        Generate auxiliary data for a single instance
        Implements Algorithm 1: Self-Refine with Polarity Intervention
        
        Args:
            instance_id: Unique instance ID
            sentence: Input sentence
            target: Target aspect term
            gold_polarity: Gold sentiment polarity
            verbose: Print debugging info
            
        Returns:
            AuxiliaryData with generated aspect/opinion and confidence
        """
        
        feedback = None
        best_aspect = ""
        best_aspect_conf = 0.0
        best_opinion = ""
        best_opinion_conf = 0.0
        converged = False
        
        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"Instance {instance_id}")
            logger.info(f"Sentence: {sentence}")
            logger.info(f"Target: {target}, Gold Polarity: {gold_polarity}")
            logger.info(f"{'='*60}")
        
        # Refinement loop
        for epoch in range(self.max_refinement_epochs):
            if verbose:
                logger.info(f"\n--- Iteration {epoch + 1} ---")
            
            # Step 1: Extract aspect
            aspect, aspect_conf = self.extractor.extract_aspect(
                sentence, target, feedback
            )
            
            if not aspect:
                if verbose:
                    logger.warning("Failed to extract aspect, skipping iteration")
                continue
            
            if verbose:
                logger.info(f"Aspect: '{aspect}' (confidence: {aspect_conf:.2f})")
            
            # Step 2: Extract opinion
            opinion, opinion_conf = self.extractor.extract_opinion(
                sentence, target, aspect, feedback
            )
            
            if not opinion:
                if verbose:
                    logger.warning("Failed to extract opinion, skipping iteration")
                continue
            
            if verbose:
                logger.info(f"Opinion: '{opinion}' (confidence: {opinion_conf:.2f})")
            
            # Step 3: Infer polarity to check against gold
            predicted_polarity, polarity_conf = self.extractor.infer_polarity(
                sentence, target, aspect, opinion
            )
            
            if not predicted_polarity:
                if verbose:
                    logger.warning("Failed to infer polarity, skipping iteration")
                continue
            
            if verbose:
                logger.info(f"Predicted polarity: {predicted_polarity} "
                           f"(confidence: {polarity_conf:.2f})")
            
            # Store best so far
            best_aspect = aspect
            best_aspect_conf = aspect_conf
            best_opinion = opinion
            best_opinion_conf = opinion_conf
            
            # Step 4: Check polarity match
            if predicted_polarity.lower() == gold_polarity.lower():
                if verbose:
                    logger.info(f"✓ Polarity match! Converged at iteration {epoch + 1}")
                converged = True
                break
            
            # Step 5: If no match, generate feedback
            if verbose:
                logger.info(f"✗ Polarity mismatch ({predicted_polarity} != {gold_polarity})")
                logger.info(f"Generating feedback...")
            
            feedback = self.extractor.generate_feedback(
                sentence, target, aspect, opinion,
                predicted_polarity, gold_polarity
            )
            
            if verbose:
                logger.info(f"Feedback: {feedback}")
        
        # Create result
        result = AuxiliaryData(
            instance_id=instance_id,
            sentence=sentence,
            target=target,
            gold_polarity=gold_polarity,
            aspect=best_aspect,
            aspect_confidence=best_aspect_conf,
            opinion=best_opinion,
            opinion_confidence=best_opinion_conf,
            refinement_iterations=epoch + 1,
            converged=converged
        )
        
        if verbose:
            logger.info(f"\nFinal result:")
            logger.info(f"  Aspect: '{result.aspect}' ({result.aspect_confidence:.2f})")
            logger.info(f"  Opinion: '{result.opinion}' ({result.opinion_confidence:.2f})")
            logger.info(f"  Converged: {result.converged}")
            logger.info(f"  Iterations: {result.refinement_iterations}")
        
        return result
    
    def generate_batch(
        self,
        instances: List[Dict],
        verbose: bool = False,
        save_interval: int = 10
    ) -> List[AuxiliaryData]:
        """
        Generate auxiliary data for batch of instances
        
        Args:
            instances: List of instance dicts with keys:
                       [id, sentence, target, gold_polarity]
            verbose: Print debugging info
            save_interval: Save progress every N instances
            
        Returns:
            List of AuxiliaryData
        """
        results = []
        
        logger.info(f"\nGenerating auxiliary data for {len(instances)} instances")
        logger.info(f"Max refinement epochs: {self.max_refinement_epochs}")
        
        pbar = tqdm(instances, desc="Generating auxiliary data")
        
        for idx, instance in enumerate(pbar):
            result = self.generate_for_instance(
                instance_id=instance['id'],
                sentence=instance['sentence'],
                target=instance['target'],
                gold_polarity=instance['gold_polarity'],
                verbose=verbose
            )
            results.append(result)
            
            # Periodic logging
            if (idx + 1) % save_interval == 0:
                converged_count = sum(1 for r in results if r.converged)
                avg_iterations = sum(r.refinement_iterations for r in results) / len(results)
                logger.info(f"\nProgress: {idx + 1}/{len(instances)}")
                logger.info(f"  Converged: {converged_count}/{len(results)} "
                           f"({100*converged_count/len(results):.1f}%)")
                logger.info(f"  Avg iterations: {avg_iterations:.2f}")
                logger.info(f"  Avg aspect confidence: "
                           f"{sum(r.aspect_confidence for r in results) / len(results):.2f}")
                logger.info(f"  Avg opinion confidence: "
                           f"{sum(r.opinion_confidence for r in results) / len(results):.2f}")
        
        return results
    
    @staticmethod
    def save_results(results: List[AuxiliaryData], output_path: str):
        """Save results to JSON"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = [r.to_dict() for r in results]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"\nSaved {len(data)} results to {output_path}")
    
    @staticmethod
    def load_results(json_path: str) -> List[AuxiliaryData]:
        """Load results from JSON"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        results = [AuxiliaryData(**item) for item in data]
        return results
    
    @staticmethod
    def print_statistics(results: List[AuxiliaryData]):
        """Print statistics about generation results"""
        total = len(results)
        converged = sum(1 for r in results if r.converged)
        
        aspect_confs = [r.aspect_confidence for r in results]
        opinion_confs = [r.opinion_confidence for r in results]
        iterations = [r.refinement_iterations for r in results]
        
        print("\n" + "=" * 60)
        print("AUXILIARY DATA GENERATION STATISTICS")
        print("=" * 60)
        print(f"Total instances: {total}")
        print(f"Converged: {converged} ({100*converged/total:.1f}%)")
        print(f"\nRefinement iterations:")
        print(f"  Mean: {sum(iterations)/len(iterations):.2f}")
        print(f"  Min: {min(iterations)}")
        print(f"  Max: {max(iterations)}")
        print(f"\nAspect confidence:")
        print(f"  Mean: {sum(aspect_confs)/len(aspect_confs):.2f}")
        print(f"  Min: {min(aspect_confs):.2f}")
        print(f"  Max: {max(aspect_confs):.2f}")
        print(f"\nOpinion confidence:")
        print(f"  Mean: {sum(opinion_confs)/len(opinion_confs):.2f}")
        print(f"  Min: {min(opinion_confs):.2f}")
        print(f"  Max: {max(opinion_confs):.2f}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Generate auxiliary data using Ollama LLM'
    )
    parser.add_argument('--input', type=str, default='data/processed/train_implicit.json',
                        help='Input JSON file with instances')
    parser.add_argument('--output', type=str, default='data/auxiliary/train_implicit_aux.json',
                        help='Output JSON file for auxiliary data')
    parser.add_argument('--model', type=str, default='mistral',
                        help='Ollama model to use')
    parser.add_argument('--max-epochs', type=int, default=10,
                        help='Maximum refinement epochs')
    parser.add_argument('--max-instances', type=int, default=None,
                        help='Maximum instances to process (for testing)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed logs')
    
    args = parser.parse_args()
    
    # Load instances
    logger.info(f"Loading instances from {args.input}")
    with open(args.input, 'r') as f:
        all_instances = json.load(f)
    
    # Limit instances if specified
    if args.max_instances:
        all_instances = all_instances[:args.max_instances]
        logger.info(f"Using first {args.max_instances} instances")
    
    # Prepare instances for generation
    instances = []
    for inst in all_instances:
        instances.append({
        'id': inst['id'],
        'sentence': inst['sentence'],
        'target': inst['target'],
        'gold_polarity': inst['gold_polarity']
    })
    
    # Initialize Ollama client
    logger.info(f"Initializing Ollama client with model: {args.model}")
    try:
        client = OllamaClient(model=args.model)
    except Exception as e:
        logger.error(f"Failed to initialize Ollama client: {e}")
        logger.error("Make sure Ollama is running: ollama serve")
        return
    
    # Initialize generator
    generator = AuxiliaryGenerator(
        ollama_client=client,
        max_refinement_epochs=args.max_epochs
    )
    
    # Generate auxiliary data
    results = generator.generate_batch(
        instances=instances,
        verbose=args.verbose,
        save_interval=5
    )
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    generator.save_results(results, args.output)
    
    # Print statistics
    generator.print_statistics(results)


if __name__ == '__main__':
    main()
