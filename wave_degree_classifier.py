"""
Elliott Wave Degree Classifier - Production Implementation
==========================================================

Intelligent wave degree classification system based on:
- Timeframe analysis
- Duration of price movement
- Price range magnitude
- Nested wave context
- Technical indicator confluence

Author: AI Financial Systems
Version: 1.0
Date: November 2024
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import re


class WaveDegreeClassifier:
    """
    Professional-grade Elliott Wave degree classification system.
    
    This classifier determines the appropriate Elliott Wave degree label
    based on multiple factors including timeframe, duration, price movement,
    and technical context.
    """
    
    # Wave degree hierarchy (higher number = larger degree)
    DEGREE_HIERARCHY = {
        1: "Grand Supercycle",
        2: "Supercycle", 
        3: "Cycle",
        4: "Primary",
        5: "Intermediate",
        6: "Minor",
        7: "Minute",
        8: "Minuette",
        9: "Subminuette",
        10: "Sub-Subminuette"
    }
    
    # Timeframe to degree mapping
    TIMEFRAME_DEGREE_MAP = {
        "1m":  ["Subminuette", "Sub-Subminuette"],
        "5m":  ["Subminuette", "Minuette"],
        "15m": ["Minuette", "Minute"],
        "30m": ["Minuette", "Minute"],
        "1h":  ["Minute", "Minor"],
        "2h":  ["Minute", "Minor"],
        "4h":  ["Minor", "Intermediate"],
        "6h":  ["Minor", "Intermediate"],
        "8h":  ["Minor", "Intermediate"],
        "Daily": ["Intermediate", "Primary"],
        "Weekly": ["Primary", "Cycle"],
        "Monthly": ["Cycle", "Supercycle"]
    }
    
    # Elliott Wave notation systems
    NOTATION_SYSTEM = {
        "Grand Supercycle": {
            "impulse": "𝐈 𝐈𝐈 𝐈𝐈𝐈 𝐈𝐕 𝐕",
            "corrective": "𝐀 𝐁 𝐂",
            "typical_use": "Multi-decade to century-long moves"
        },
        "Supercycle": {
            "impulse": "⦅I⦆ ⦅II⦆ ⦅III⦆ ⦅IV⦆ ⦅V⦆",
            "corrective": "⦅A⦆ ⦅B⦆ ⦅C⦆",
            "typical_use": "Decade-long bull/bear markets"
        },
        "Cycle": {
            "impulse": "[I] [II] [III] [IV] [V]",
            "corrective": "[A] [B] [C]",
            "typical_use": "Multi-year market cycles"
        },
        "Primary": {
            "impulse": "① ② ③ ④ ⑤",
            "corrective": "Ⓐ Ⓑ Ⓒ",
            "typical_use": "Several months to 1-2 years"
        },
        "Intermediate": {
            "impulse": "(1) (2) (3) (4) (5)",
            "corrective": "(A) (B) (C)",
            "typical_use": "Weeks to several months"
        },
        "Minor": {
            "impulse": "1 2 3 4 5",
            "corrective": "A B C",
            "typical_use": "Days to weeks"
        },
        "Minute": {
            "impulse": "i ii iii iv v",
            "corrective": "a b c",
            "typical_use": "Hours to days"
        },
        "Minuette": {
            "impulse": "(i) (ii) (iii) (iv) (v)",
            "corrective": "(a) (b) (c)",
            "typical_use": "Minutes to hours"
        },
        "Subminuette": {
            "impulse": "i ii iii iv v",
            "corrective": "a b c",
            "typical_use": "Sub-hour timeframes"
        },
        "Sub-Subminuette": {
            "impulse": "i ii iii iv v",
            "corrective": "a b c",
            "typical_use": "Minute-level subdivisions"
        }
    }
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the classifier.
        
        Args:
            verbose: If True, print detailed classification steps
        """
        self.verbose = verbose
        
    def classify(self, context: Dict) -> Dict:
        """
        Main classification method.
        
        Args:
            context: Dictionary containing:
                - symbol (str): Ticker symbol (e.g., 'PLTR', 'SPY')
                - timeframe (str): Chart timeframe (e.g., '4h', 'Daily')
                - duration (str): Duration of price move (e.g., '5 days', '3 hours')
                - price_range (str): Price range (e.g., '$82.50 to $87.45')
                - trend (str): 'uptrend', 'downtrend', or 'sideways'
                - parent_wave_degree (str, optional): Parent wave degree if nested
                - indicators (dict, optional): Technical indicator values
                
        Returns:
            Dictionary containing:
                - wave_degree (str): Classified wave degree
                - label_style (str): Impulse wave notation
                - corrective_style (str): Corrective wave notation
                - confidence (float): Confidence score 0-100
                - reasoning (str): Detailed reasoning for classification
        """
        
        # Validate required fields
        required_fields = ['symbol', 'timeframe', 'duration', 'price_range', 'trend']
        for field in required_fields:
            if field not in context:
                raise ValueError(f"Missing required field: {field}")
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"CLASSIFYING WAVE FOR {context['symbol']}")
            print(f"{'='*60}\n")
        
        # Step 1: Parse duration into hours
        duration_hours = self._parse_duration(context['duration'])
        if self.verbose:
            print(f"Duration: {duration_hours:.1f} hours ({context['duration']})")
        
        # Step 2: Calculate price movement percentage
        price_move_pct = self._calculate_price_move(context['price_range'])
        if self.verbose:
            print(f"Price Movement: {price_move_pct:.2f}% ({context['price_range']})")
        
        # Step 3: Get base degrees from timeframe
        base_degrees = self.TIMEFRAME_DEGREE_MAP.get(
            context['timeframe'], 
            ["Minor"]
        )
        if self.verbose:
            print(f"Base Degrees (from {context['timeframe']}): {base_degrees}")
        
        # Step 4: Classify by duration
        duration_degree = self._classify_by_duration(duration_hours)
        if self.verbose:
            print(f"Duration-Based Degree: {duration_degree}")
        
        # Step 5: Adjust for price movement magnitude
        price_degree = self._classify_by_price_move(price_move_pct)
        if self.verbose:
            print(f"Price-Based Degree: {price_degree}")
        
        # Step 6: Consider nested context if provided
        nested_degree = None
        if 'parent_wave_degree' in context:
            nested_degree = self._get_nested_degree(context['parent_wave_degree'])
            if self.verbose:
                print(f"Nested Degree (from {context['parent_wave_degree']}): {nested_degree}")
        
        # Step 7: Synthesize final degree
        final_degree = self._synthesize_degree(
            base_degrees,
            duration_degree,
            price_degree,
            nested_degree,
            context['timeframe']
        )
        if self.verbose:
            print(f"\nFinal Degree: {final_degree}")
        
        # Step 8: Calculate confidence
        confidence = self._calculate_confidence(
            context,
            final_degree,
            base_degrees,
            duration_degree,
            price_degree
        )
        if self.verbose:
            print(f"Confidence: {confidence}%")
        
        # Step 9: Generate reasoning
        reasoning = self._generate_reasoning(
            context,
            final_degree,
            confidence,
            duration_hours,
            price_move_pct
        )
        
        # Get notation
        notation = self.NOTATION_SYSTEM[final_degree]
        
        result = {
            'wave_degree': final_degree,
            'label_style': notation['impulse'],
            'corrective_style': notation['corrective'],
            'typical_use': notation['typical_use'],
            'confidence': confidence,
            'reasoning': reasoning,
            'metadata': {
                'symbol': context['symbol'],
                'timeframe': context['timeframe'],
                'duration_hours': duration_hours,
                'price_move_pct': price_move_pct,
                'trend': context['trend'],
                'timestamp': datetime.now().isoformat()
            }
        }
        
        if self.verbose:
            print(f"\n{reasoning}")
        
        return result
    
    def _parse_duration(self, duration_str: str) -> float:
        """
        Parse duration string into hours.
        
        Args:
            duration_str: Duration string (e.g., "5 days", "3 hours", "2 weeks")
            
        Returns:
            Duration in hours as float
        """
        duration_str = duration_str.lower().strip()
        
        # Try to extract number and unit
        match = re.search(r'(\d+\.?\d*)\s*(hour|day|week|month|year)', duration_str)
        
        if not match:
            # Try just a number (assume hours)
            match = re.search(r'(\d+\.?\d*)', duration_str)
            if match:
                return float(match.group(1))
            return 24.0  # Default to 1 day
        
        value = float(match.group(1))
        unit = match.group(2)
        
        # Conversion factors to hours
        conversions = {
            'hour': 1,
            'day': 24,
            'week': 168,
            'month': 720,  # Approximate (30 days)
            'year': 8760   # 365 days
        }
        
        return value * conversions.get(unit, 1)
    
    def _calculate_price_move(self, range_str: str) -> float:
        """
        Calculate percentage price movement from range string.
        
        Args:
            range_str: Price range string (e.g., "$82.50 to $87.45")
            
        Returns:
            Percentage move as float
        """
        # Remove currency symbols and extract numbers
        numbers = re.findall(r'\d+\.?\d*', range_str)
        
        if len(numbers) < 2:
            return 5.0  # Default assumption
        
        low = float(numbers[0])
        high = float(numbers[1])
        
        # Calculate percentage change
        return ((high - low) / low) * 100
    
    def _classify_by_duration(self, hours: float) -> str:
        """
        Classify wave degree based on duration in hours.
        
        Args:
            hours: Duration in hours
            
        Returns:
            Wave degree string
        """
        if hours < 1:
            return "Sub-Subminuette"
        elif hours < 2:
            return "Subminuette"
        elif hours < 8:
            return "Minuette"
        elif hours < 24:
            return "Minute"
        elif hours < 72:  # 3 days
            return "Minor"
        elif hours < 240:  # 10 days
            return "Intermediate"
        elif hours < 720:  # 30 days
            return "Primary"
        elif hours < 2160:  # 90 days
            return "Cycle"
        else:
            return "Supercycle"
    
    def _classify_by_price_move(self, pct: float) -> str:
        """
        Classify wave degree based on price movement percentage.
        
        Args:
            pct: Price movement percentage
            
        Returns:
            Wave degree string
        """
        if pct < 0.5:
            return "Subminuette"
        elif pct < 1:
            return "Minuette"
        elif pct < 2:
            return "Minute"
        elif pct < 5:
            return "Minor"
        elif pct < 10:
            return "Intermediate"
        elif pct < 20:
            return "Primary"
        elif pct < 40:
            return "Cycle"
        else:
            return "Supercycle"
    
    def _get_nested_degree(self, parent_degree: str) -> str:
        """
        Get appropriate sub-degree for nested waves.
        
        Args:
            parent_degree: Parent wave degree
            
        Returns:
            Sub-degree (one level smaller)
        """
        # Find parent degree in hierarchy
        degree_list = list(self.DEGREE_HIERARCHY.values())
        
        if parent_degree not in degree_list:
            return "Minor"  # Default
        
        parent_index = degree_list.index(parent_degree)
        
        # Return next smaller degree
        if parent_index < len(degree_list) - 1:
            return degree_list[parent_index + 1]
        
        return parent_degree  # Already at smallest
    
    def _synthesize_degree(
        self,
        base_degrees: List[str],
        duration_degree: str,
        price_degree: str,
        nested_degree: Optional[str],
        timeframe: str
    ) -> str:
        """
        Synthesize final degree from multiple signals.
        
        Args:
            base_degrees: List of degrees from timeframe
            duration_degree: Degree from duration analysis
            price_degree: Degree from price movement
            nested_degree: Degree from parent wave (if applicable)
            timeframe: Chart timeframe
            
        Returns:
            Final wave degree
        """
        # Collect all candidates
        candidates = [base_degrees[0], duration_degree, price_degree]
        
        if nested_degree:
            candidates.append(nested_degree)
        
        # Weight by priority
        # Priority: timeframe > duration > price > nested
        weights = {}
        
        # Timeframe gets highest weight
        weights[base_degrees[0]] = weights.get(base_degrees[0], 0) + 3
        
        # Duration gets medium-high weight
        weights[duration_degree] = weights.get(duration_degree, 0) + 2
        
        # Price gets medium weight
        weights[price_degree] = weights.get(price_degree, 0) + 1.5
        
        # Nested gets lower weight (it's supplementary)
        if nested_degree:
            weights[nested_degree] = weights.get(nested_degree, 0) + 1
        
        # Find degree with highest weight
        final_degree = max(weights.items(), key=lambda x: x[1])[0]
        
        # Sanity check: ensure degree makes sense for timeframe
        degree_list = list(self.DEGREE_HIERARCHY.values())
        final_index = degree_list.index(final_degree)
        base_index = degree_list.index(base_degrees[0])
        
        # If final is more than 2 degrees away from base, use base
        if abs(final_index - base_index) > 2:
            final_degree = base_degrees[0]
        
        return final_degree
    
    def _calculate_confidence(
        self,
        context: Dict,
        final_degree: str,
        base_degrees: List[str],
        duration_degree: str,
        price_degree: str
    ) -> float:
        """
        Calculate confidence score for classification.
        
        Args:
            context: Original context dictionary
            final_degree: Classified wave degree
            base_degrees: Degrees from timeframe
            duration_degree: Degree from duration
            price_degree: Degree from price
            
        Returns:
            Confidence score 0-100
        """
        base_confidence = 75.0
        
        # If all signals agree, boost confidence
        degrees = [base_degrees[0], duration_degree, price_degree]
        if all(d == final_degree for d in degrees):
            base_confidence += 15
        
        # If most signals agree (2 out of 3)
        elif degrees.count(final_degree) >= 2:
            base_confidence += 8
        
        # Trend clarity adjustment
        if context['trend'] == 'sideways':
            base_confidence -= 12
        elif context['trend'] in ['uptrend', 'downtrend']:
            base_confidence += 5
        
        # Indicator confluence adjustment
        if 'indicators' in context and context['indicators']:
            indicator_count = len(context['indicators'])
            base_confidence += min(indicator_count * 2, 10)
        
        # Parent wave context adjustment
        if 'parent_wave_degree' in context:
            base_confidence += 5
        
        # Cap between 50 and 98
        return min(max(base_confidence, 50.0), 98.0)
    
    def _generate_reasoning(
        self,
        context: Dict,
        final_degree: str,
        confidence: float,
        duration_hours: float,
        price_move_pct: float
    ) -> str:
        """
        Generate human-readable reasoning for classification.
        
        Args:
            context: Original context
            final_degree: Classified degree
            confidence: Confidence score
            duration_hours: Parsed duration
            price_move_pct: Calculated price movement
            
        Returns:
            Multi-line reasoning string
        """
        lines = []
        
        lines.append(f"ELLIOTT WAVE DEGREE CLASSIFICATION")
        lines.append(f"{'='*50}")
        lines.append(f"")
        lines.append(f"Symbol: {context['symbol']}")
        lines.append(f"Classified Degree: {final_degree}")
        lines.append(f"Confidence: {confidence:.1f}%")
        lines.append(f"")
        lines.append(f"ANALYSIS FACTORS:")
        lines.append(f"")
        lines.append(f"✓ Timeframe: {context['timeframe']}")
        lines.append(f"  → Typical degrees: {', '.join(self.TIMEFRAME_DEGREE_MAP.get(context['timeframe'], ['Minor']))}")
        lines.append(f"")
        lines.append(f"✓ Duration: {duration_hours:.1f} hours ({context['duration']})")
        lines.append(f"  → Duration suggests: {self._classify_by_duration(duration_hours)}")
        lines.append(f"")
        lines.append(f"✓ Price Movement: {price_move_pct:.2f}% ({context['price_range']})")
        lines.append(f"  → Magnitude suggests: {self._classify_by_price_move(price_move_pct)}")
        lines.append(f"")
        lines.append(f"✓ Trend Direction: {context['trend'].upper()}")
        
        if 'parent_wave_degree' in context:
            lines.append(f"")
            lines.append(f"✓ Parent Wave Degree: {context['parent_wave_degree']}")
            lines.append(f"  → Nested structure confirmed")
        
        if 'indicators' in context and context['indicators']:
            lines.append(f"")
            lines.append(f"✓ Technical Indicators:")
            for indicator, value in context['indicators'].items():
                lines.append(f"  - {indicator.upper()}: {value}")
        
        lines.append(f"")
        lines.append(f"NOTATION TO USE:")
        lines.append(f"")
        notation = self.NOTATION_SYSTEM[final_degree]
        lines.append(f"• Impulse Waves: {notation['impulse']}")
        lines.append(f"• Corrective Waves: {notation['corrective']}")
        lines.append(f"• Typical Context: {notation['typical_use']}")
        lines.append(f"")
        lines.append(f"RECOMMENDATION:")
        lines.append(f"Label this wave structure using {final_degree} degree notation.")
        lines.append(f"This classification is based on the combination of timeframe,")
        lines.append(f"duration, and price movement characteristics.")
        
        return '\n'.join(lines)
    
    def classify_multiple_timeframes(
        self,
        symbol: str,
        timeframes: List[str],
        base_context: Dict
    ) -> Dict[str, Dict]:
        """
        Classify wave degrees across multiple timeframes.
        
        Args:
            symbol: Ticker symbol
            timeframes: List of timeframes to analyze
            base_context: Base context dict (without timeframe)
            
        Returns:
            Dictionary mapping timeframe to classification result
        """
        results = {}
        
        for timeframe in timeframes:
            context = base_context.copy()
            context['symbol'] = symbol
            context['timeframe'] = timeframe
            
            results[timeframe] = self.classify(context)
        
        return results


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*70)
    print("ELLIOTT WAVE DEGREE CLASSIFIER - DEMONSTRATION")
    print("="*70 + "\n")
    
    # Initialize classifier with verbose output
    classifier = WaveDegreeClassifier(verbose=True)
    
    # Example 1: PLTR on 4h chart
    print("\n" + "="*70)
    print("EXAMPLE 1: PLTR Wave 3 Analysis")
    print("="*70)
    
    pltr_context = {
        'symbol': 'PLTR',
        'timeframe': '4h',
        'duration': '5 days',
        'price_range': '$79.85 to $87.45',
        'trend': 'uptrend',
        'indicators': {
            'rsi': 68,
            'macd': 'bullish crossover',
            'volume': 'declining on advance'
        }
    }
    
    result = classifier.classify(pltr_context)
    
    print(f"\n{'='*70}")
    print("CLASSIFICATION COMPLETE")
    print(f"{'='*70}")
    print(f"\nResult JSON:")
    print(json.dumps(result, indent=2))
    
    # Example 2: SPY on Daily chart
    print("\n\n" + "="*70)
    print("EXAMPLE 2: SPY Wave Analysis")
    print("="*70)
    
    spy_context = {
        'symbol': 'SPY',
        'timeframe': 'Daily',
        'duration': '3 weeks',
        'price_range': '$540 to $595',
        'trend': 'uptrend',
        'parent_wave_degree': 'Primary'
    }
    
    result2 = classifier.classify(spy_context)
    
    print(f"\n{'='*70}")
    print("CLASSIFICATION COMPLETE")
    print(f"{'='*70}")
    
    # Example 3: Multi-timeframe analysis
    print("\n\n" + "="*70)
    print("EXAMPLE 3: Multi-Timeframe Classification")
    print("="*70)
    
    base_context = {
        'duration': '4 days',
        'price_range': '$100 to $108',
        'trend': 'uptrend'
    }
    
    classifier_quiet = WaveDegreeClassifier(verbose=False)
    mtf_results = classifier_quiet.classify_multiple_timeframes(
        symbol='QQQ',
        timeframes=['1h', '4h', 'Daily', 'Weekly'],
        base_context=base_context
    )
    
    print("\nMulti-Timeframe Results:")
    print("-" * 70)
    for tf, result in mtf_results.items():
        print(f"\n{tf:8s} → {result['wave_degree']:15s} "
              f"({result['confidence']:.0f}% confidence)")
        print(f"         Notation: {result['label_style']}")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70 + "\n")
