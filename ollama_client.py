"""
Ollama Client for Local LLM Integration
Connects to Ollama server and handles LLM-based auxiliary task generation
"""

import requests
import json
import time
from typing import Tuple, Dict, Optional
import logging
from dataclasses import dataclass


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Response from LLM"""
    text: str
    confidence: float
    raw_response: Optional[str] = None


class OllamaClient:
    """
    Client for Ollama local LLM
    
    Ollama should be running on http://localhost:11434 (default)
    
    Usage:
        client = OllamaClient(model='mistral')
        response = client.generate("What is sentiment analysis?")
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "mistral",
        temperature: float = 0.7,
        max_tokens: int = 200,
        timeout: int = 120
    ):
        """
        Initialize Ollama client
        
        Args:
            base_url: URL where Ollama is running
            model: Model name (must be pulled first with 'ollama pull <model>')
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        self._verify_connection()
        self._verify_model()
    
    def _verify_connection(self):
        """Check if Ollama is running"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            if response.status_code != 200:
                raise ConnectionError(f"Ollama returned status {response.status_code}")
            logger.info("✓ Connected to Ollama")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure to run: ollama serve"
            )
        except Exception as e:
            raise ConnectionError(f"Error connecting to Ollama: {e}")
    
    def _verify_model(self):
        """Check if model is available"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if not any(self.model in name for name in model_names):
                raise ValueError(
                    f"Model '{self.model}' not found. Available models: {model_names}. "
                    f"Pull model with: ollama pull {self.model}"
                )
            
            logger.info(f"✓ Model '{self.model}' available")
        except Exception as e:
            raise ValueError(f"Error checking model: {e}")
    
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> LLMResponse:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            temperature: Optional override for temperature
            max_tokens: Optional override for max tokens
            
        Returns:
            LLMResponse with generated text and confidence
        """
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                logger.error(f"LLM Error: {response.text}")
                return LLMResponse(text="", confidence=0.0)
            
            result = response.json()
            generated_text = result.get('response', '').strip()
            
            # Estimate confidence from response
            confidence = self._estimate_confidence(
                prompt,
                generated_text,
                result
            )
            
            return LLMResponse(
                text=generated_text,
                confidence=confidence,
                raw_response=json.dumps(result)
            )
        
        except requests.exceptions.Timeout:
            logger.error(f"LLM request timed out after {self.timeout}s")
            return LLMResponse(text="", confidence=0.0)
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return LLMResponse(text="", confidence=0.0)
    
    def _estimate_confidence(
        self,
        prompt: str,
        response: str,
        llm_result: Dict
    ) -> float:
        """
        Estimate confidence in the response
        
        Based on:
        1. Response length (too short/long = lower confidence)
        2. Response coherence (based on perplexity if available)
        3. Presence of uncertainty markers
        
        Returns value between 0.5 and 1.0
        """
        if not response:
            return 0.5
        
        confidence = 0.75  # Base confidence
        
        # Penalty for very short responses
        if len(response.split()) < 2:
            confidence -= 0.15
        
        # Penalty for very long responses
        if len(response.split()) > 50:
            confidence -= 0.10
        
        # Penalty for uncertainty markers
        uncertainty_phrases = [
            "i don't know", "i'm not sure", "possibly", "maybe", 
            "might be", "could be", "uncertain", "unclear"
        ]
        if any(phrase in response.lower() for phrase in uncertainty_phrases):
            confidence -= 0.15
        
        # Penalty for incomplete responses
        if response.endswith("...") or response.endswith("?") or len(response) > 95:
            confidence -= 0.10
        
        # Ensure confidence stays in [0.5, 1.0]
        return max(0.5, min(1.0, confidence))
    
    def extract_json(self, response_text: str) -> Optional[Dict]:
        """
        Extract JSON from model response
        Attempts to parse JSON from the response text
        """
        try:
            # Try direct parse
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Try finding JSON in the response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            if start != -1 and end > start:
                try:
                    return json.loads(response_text[start:end])
                except:
                    pass
        
        return None


class AspectOpinionExtractor:
    """
    Use Ollama to extract aspects and opinions
    Implements the auxiliary task generation with confidence scores
    """
    
    def __init__(self, ollama_client: OllamaClient):
        self.client = ollama_client
    
    def extract_aspect(
        self,
        sentence: str,
        target: str,
        feedback: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Extract implicit aspect from sentence
        
        Args:
            sentence: Input sentence
            target: Target aspect term
            feedback: Optional feedback from previous iteration
            
        Returns:
            (aspect_text, confidence)
        """
        if feedback:
            prompt = f"""Given the sentence: "{sentence}"
Target aspect: "{target}"
Previous feedback: {feedback}

Based on the feedback, what is the MOST RELEVANT implicit aspect element related to '{target}'?
Provide ONLY the aspect phrase (1-3 words), nothing else."""
        else:
            prompt = f"""Given the sentence: "{sentence}"
Target aspect: "{target}"

What is the most relevant implicit aspect element or dimension related to '{target}' in this sentence?
Provide ONLY the aspect phrase (1-3 words), nothing else."""
        
        response = self.client.generate(prompt, temperature=0.5)
        
        # Clean response
        aspect_text = response.text.strip().strip('*').strip('"').strip()
        
        # Filter out junk responses
        if not aspect_text or len(aspect_text.split()) > 5 or aspect_text.lower() == "null":
            return "", 0.5
        
        return aspect_text, response.confidence
    
    def extract_opinion(
        self,
        sentence: str,
        target: str,
        aspect: str,
        feedback: Optional[str] = None
    ) -> Tuple[str, float]:
        """
        Extract opinion towards the aspect
        
        Args:
            sentence: Input sentence
            target: Target aspect term
            aspect: Extracted aspect element
            feedback: Optional feedback from previous iteration
            
        Returns:
            (opinion_text, confidence)
        """
        if feedback:
            prompt = f"""Given the sentence: "{sentence}"
Target aspect: "{target}"
Aspect element: "{aspect}"
Previous feedback: {feedback}

Based on the feedback, what is the opinion towards the '{aspect}'?
Provide ONLY the opinion phrase (1-3 words), nothing else."""
        else:
            prompt = f"""Given the sentence: "{sentence}"
Target aspect: "{target}"
Aspect element: "{aspect}"

What is the opinion or evaluation towards the '{aspect}'?
Provide ONLY the opinion phrase (1-3 words), nothing else."""
        
        response = self.client.generate(prompt, temperature=0.5)
        
        opinion_text = response.text.strip().strip('*').strip('"').strip()
        
        # Filter junk responses
        if not opinion_text or len(opinion_text.split()) > 5 or opinion_text.lower() == "null":
            return "", 0.5
        
        return opinion_text, response.confidence
    
    def infer_polarity(
        self,
        sentence: str,
        target: str,
        aspect: str,
        opinion: str
    ) -> Tuple[str, float]:
        """
        Infer polarity from aspect and opinion
        
        Args:
            sentence: Input sentence
            target: Target aspect term
            aspect: Aspect element
            opinion: Opinion text
            
        Returns:
            (polarity, confidence)
        """
        prompt = f"""Given:
Sentence: "{sentence}"
Target: "{target}"
Aspect: "{aspect}"
Opinion: "{opinion}"

What is the sentiment polarity (positive, negative, or neutral)?
Answer with ONLY one word: positive, negative, or neutral"""
        
        response = self.client.generate(prompt, temperature=0.3)
        polarity = response.text.strip().lower()
        
        # Validate polarity
        valid_polarities = ['positive', 'negative', 'neutral']
        if polarity not in valid_polarities:
            return "", 0.5
        
        return polarity, response.confidence
    
    def generate_feedback(
        self,
        sentence: str,
        target: str,
        aspect: str,
        opinion: str,
        predicted_polarity: str,
        gold_polarity: str
    ) -> str:
        """
        Generate feedback for refinement when polarity doesn't match
        (when opinion was successfully extracted)
        
        Args:
            sentence: Input sentence
            target: Target aspect term
            aspect: Current aspect extraction
            opinion: Current opinion extraction
            predicted_polarity: Model's predicted polarity
            gold_polarity: Ground truth polarity
            
        Returns:
            Feedback string for next iteration
        """
        prompt = f"""The aspect-opinion extraction failed to match the gold label.

Sentence: "{sentence}"
Target: "{target}"
Current aspect: "{aspect}"
Current opinion: "{opinion}"
Predicted polarity: {predicted_polarity}
Gold polarity: {gold_polarity}

Provide ONE sentence of feedback to guide the next extraction attempt.
Focus on what aspect or dimension should be emphasized instead."""
        
        response = self.client.generate(prompt, temperature=0.7, max_tokens=50)
        return response.text.strip()
    
    def infer_polarity_from_sentence(
        self,
        sentence: str,
        target: str,
        aspect: str
    ) -> Tuple[str, float]:
        """
        IMPROVED: Infer polarity when opinion extraction failed.
        Uses aspect + sentence directly instead of aspect + opinion.
        
        Args:
            sentence: Input sentence
            target: Target aspect term
            aspect: Extracted aspect element
            
        Returns:
            (polarity, confidence)
        """
        prompt = f"""Given:
Sentence: "{sentence}"
Target: "{target}"
Aspect: "{aspect}"

What is the sentiment polarity towards the aspect (positive, negative, or neutral)?
Consider the entire sentence context since there is no explicit opinion term.
Answer with ONLY one word: positive, negative, or neutral"""
        
        response = self.client.generate(prompt, temperature=0.3)
        polarity = response.text.strip().lower()
        
        # Validate polarity
        valid_polarities = ['positive', 'negative', 'neutral']
        if polarity not in valid_polarities:
            return "", 0.5
        
        return polarity, response.confidence
    
    def generate_feedback_no_opinion(
        self,
        sentence: str,
        target: str,
        aspect: str,
        predicted_polarity: str,
        gold_polarity: str
    ) -> str:
        """
        IMPROVED: Generate feedback when opinion extraction failed.
        Focuses on aspect refinement since opinion wasn't available.
        
        Args:
            sentence: Input sentence
            target: Target aspect term
            aspect: Current aspect extraction
            predicted_polarity: Model's predicted polarity
            gold_polarity: Ground truth polarity
            
        Returns:
            Feedback string for next iteration
        """
        prompt = f"""The aspect extraction and polarity inference failed to match the gold label.
(Note: No opinion term was found in this sentence)

Sentence: "{sentence}"
Target: "{target}"
Current aspect: "{aspect}"
Predicted polarity: {predicted_polarity}
Gold polarity: {gold_polarity}

Provide ONE sentence of feedback to refine the aspect extraction.
Focus on what dimension or characteristic of the aspect should be emphasized instead."""
        
        response = self.client.generate(prompt, temperature=0.7, max_tokens=50)
        return response.text.strip()


def test_ollama_connection():
    """Test script to verify Ollama connection"""
    print("Testing Ollama connection...")
    
    try:
        client = OllamaClient(model='mistral')
        print("\nGenerating test response...")
        
        response = client.generate(
            "What is implicit sentiment analysis? (answer in one sentence)"
        )
        
        print(f"Response: {response.text}")
        print(f"Confidence: {response.confidence:.2f}")
        print("\n✓ Ollama client working!")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is installed: https://ollama.ai/")
        print("2. Start Ollama: ollama serve")
        print("3. Pull a model: ollama pull mistral")
        print("4. Verify it's running: curl http://localhost:11434/api/tags")


if __name__ == '__main__':
    test_ollama_connection()
