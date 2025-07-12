#!/usr/bin/env python3
"""
HELIX NLCA Edition
Dialectical AI system that maintains productive disagreement
Truth vs Comfort cores with NLCA synthesis and Krishna metrics
Created by Jacob Kemp - Rebuilt with advanced architecture
"""

import asyncio
import httpx
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import os
from dotenv import load_dotenv
import tiktoken
from sentence_transformers import SentenceTransformer
from neo4j import AsyncGraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
import re
from utils import try_extract_json


# Load environment variables
load_dotenv()

# ==========================================
# CONFIGURATION
# ==========================================

HELIX_CONFIG = {
    'cores': {
        'truth': {
            'api': 'anthropic',
            'model': 'claude-3-5-sonnet-20241022',
            'endpoint': 'https://api.anthropic.com/v1/messages',
            'temperature': 0.3,
            'max_tokens': 500,
            'identity': """…your truth identity…"""
        },
        'comfort': {
            'api': 'openai',
            'model': 'gpt-4-turbo-preview',
            'endpoint': 'https://api.openai.com/v1/chat/completions',
            'temperature': 0.7,
            'max_tokens': 500,
            'identity': """…your comfort identity…"""
        }
    },
    'synthesis': {
        # use Mistral via Ollama HTTP generate API
        'api': 'ollama',
        'model': os.getenv('SYNTHESIS_MODEL', 'mistral:instruct'),
        'endpoint': os.getenv(
            'SYNTHESIS_ENDPOINT',
            'http://localhost:11435/api/generate'
        ),
        'max_loops': 6,
        'temperature': 0.5,
        'convergence_threshold': 0.1,
        'preserve_tension': True
    },
    'neo4j': {
        'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
        'user': os.getenv('NEO4J_USER', 'neo4j'),
        'password': os.getenv('NEO4J_PASSWORD', 'helix-knowledge')
    },
    'embeddings': {
        'model': 'all-MiniLM-L6-v2',
        'dimension': 384
    },
    'api_keys': {
        'anthropic': os.getenv('ANTHROPIC_API_KEY'),
        'openai':   os.getenv('OPENAI_API_KEY')
    }
}
# Paths
HELIX_HOME = Path.home() / "helix_nlca"
HELIX_HOME.mkdir(exist_ok=True)
LOG_PATH = HELIX_HOME / "helix_nlca.log"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

import re
import json
import logging

def parse_plain_response(text):
    pattern = r"Feature agreement:\s*([\d.]+)\s*Sign agreement:\s*([\d.]+)\s*Tension themes:\s*(\d+)"
    match = re.search(pattern, text, re.MULTILINE)
    if match:
        return {
            "feature_agreement": float(match.group(1)),
            "sign_agreement": float(match.group(2)),
            "tension_themes": int(match.group(3))
        }
    return None

# ==========================================
# DATA STRUCTURES - ADD ThoughtSynapse
# ==========================================

@dataclass
class ThoughtSynapse:
    """Represents a thought fragment in the mesh"""
    source: str
    content: str
    embedding: Optional[np.ndarray]
    timestamp: float
    confidence: float
    tokens_used: int
    cost: float
    round: int = 0
    responding_to: Optional[str] = None

@dataclass
class CoreResponse:
    """Response from a core with assessment"""
    core_name: str
    content: str
    assessment: Dict[str, Any]
    tokens_used: int
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DebateRound:
    """A round of debate between cores"""
    round_number: int
    truth_response: CoreResponse
    comfort_response: CoreResponse

@dataclass
class KrishnaMetrics:
    """Disagreement metrics between cores"""
    feature_agreement: float
    rank_agreement: float
    sign_agreement: float
    shared_themes: List[str]
    tension_themes: List[Dict[str, Any]]  # Themes with high importance but opposite valence

@dataclass
class Synthesis:
    """Synthesis result with tension preservation"""
    content: str
    tension_score: float
    loops_completed: int
    preserved_tensions: List[str]
    tokens_used: int

@dataclass
class ConversationState:
    """Tracks conversation state"""
    user_name: str = "User"
    turn_count: int = 0
    current_themes: List[str] = field(default_factory=list)
    tension_patterns: List[Dict[str, Any]] = field(default_factory=list)


# ==========================================
# NEURAL MESH - SHORT-TERM MEMORY
# ==========================================

class NeuralMesh:
    """Local model acting as cognitive substrate - tracks patterns in real-time"""

    def __init__(self):
        self.thought_buffer = deque(maxlen=100)
        self.debate_history = []
        self.contradiction_patterns = defaultdict(int)
        self.convergence_patterns = defaultdict(int)

    async def process_synapse(self, synapse: ThoughtSynapse) -> Dict[str, Any]:
        """Process thought through local mesh"""

        # Update thought buffer
        self.thought_buffer.append(synapse)

        # Track debate patterns
        if synapse.round > 0:
            self._track_debate_patterns(synapse)

        # Calculate mesh state
        coherence = self.calculate_coherence()
        tension = self.calculate_tension()

        return {
            'coherence': coherence,
            'tension': tension,
            'pattern': self.detect_thought_pattern(),
            'mesh_state': 'exploring' if tension > 0.5 else 'converging'
        }

    def _track_debate_patterns(self, synapse: ThoughtSynapse):
        """Track patterns in debates using linguistic markers"""
        if synapse.responding_to:
            # Look for contradiction markers
            contradiction_markers = ['however', 'but', 'actually', 'disagree', 'wrong', 'no,', 'problematic']
            if any(marker in synapse.content.lower() for marker in contradiction_markers):
                self.contradiction_patterns[synapse.source] += 1

            # Look for convergence markers
            convergence_markers = ['agree', 'yes', 'exactly', 'both true', 'valid point', 'that resonates']
            if any(marker in synapse.content.lower() for marker in convergence_markers):
                self.convergence_patterns[synapse.source] += 1

    def calculate_coherence(self) -> float:
        """Measure coherence of recent thoughts"""
        if len(self.thought_buffer) < 2:
            return 1.0

        recent = list(self.thought_buffer)[-5:]
        if len(recent) < 2:
            return 1.0

        coherence_scores = []
        for i in range(len(recent) - 1):
            if recent[i].embedding is not None and recent[i+1].embedding is not None:
                if recent[i].embedding.shape == recent[i+1].embedding.shape:
                    sim = cosine_similarity([recent[i].embedding], [recent[i+1].embedding])[0][0]
                    coherence_scores.append(sim)

        return np.mean(coherence_scores) if coherence_scores else 0.5

    def calculate_tension(self) -> float:
        """Measure productive tension between viewpoints"""
        total_patterns = sum(self.contradiction_patterns.values()) + sum(self.convergence_patterns.values())
        if total_patterns == 0:
            return 0.5

        contradiction_ratio = sum(self.contradiction_patterns.values()) / total_patterns
        return contradiction_ratio

    def detect_thought_pattern(self) -> str:
        """Detect current thought pattern"""
        if len(self.thought_buffer) < 3:
            return 'initializing'

        recent = list(self.thought_buffer)[-3:]

        # Check for debate
        if any(s.round > 0 for s in recent):
            return 'debating'

        # Check for convergence
        if all(s.source == recent[0].source for s in recent):
            return 'converging'

        return 'exploring'

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of current patterns"""
        return {
            'buffer_size': len(self.thought_buffer),
            'total_contradictions': sum(self.contradiction_patterns.values()),
            'total_convergences': sum(self.convergence_patterns.values()),
            'tension': self.calculate_tension(),
            'coherence': self.calculate_coherence(),
            'pattern': self.detect_thought_pattern()
        }

# ==========================================
# NEO4J INTERFACE
# ==========================================

class Neo4jConnection:
    """Async Neo4j connection manager"""

    def __init__(self, config: Dict[str, str]):
        self.driver = AsyncGraphDatabase.driver(
            config['uri'],
            auth=(config['user'], config['password'])
        )

    async def close(self):
        await self.driver.close()

    async def execute_query(self, query: str, parameters: Dict[str, Any] = None) -> List[Dict]:
        """Execute a Cypher query"""
        async with self.driver.session() as session:
            result = await session.run(query, parameters or {})
            return [record.data() async for record in result]

    async def initialize_schema(self):
        """Create indexes and constraints"""
        queries = [
            # Constraints
            "CREATE CONSTRAINT conversation_id IF NOT EXISTS FOR (c:Conversation) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT theme_name IF NOT EXISTS FOR (t:Theme) REQUIRE t.name IS UNIQUE",

            # Indexes
            "CREATE INDEX theme_embedding IF NOT EXISTS FOR (t:Theme) ON (t.embedding)",
            "CREATE INDEX response_timestamp IF NOT EXISTS FOR (r:Response) ON (r.timestamp)",
            "CREATE INDEX turn_number IF NOT EXISTS FOR (t:Turn) ON (t.number)"
        ]

        for query in queries:
            try:
                await self.execute_query(query)
            except Exception as e:
                logging.debug(f"Schema creation: {e}")

# ==========================================
# EMBEDDING MANAGER
# ==========================================

class EmbeddingManager:
    """Manages Sentence-BERT embeddings"""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode(self, text: str) -> np.ndarray:
        """Encode text to embedding"""
        return self.model.encode(text, convert_to_tensor=False)

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """Encode multiple texts"""
        return self.model.encode(texts, convert_to_tensor=False)

# ==========================================
# API MANAGER
# ==========================================

class APIManager:
    """Manages async API calls to various providers"""

    def __init__(self):
        self.token_encoder = tiktoken.encoding_for_model("gpt-4")
        self.total_tokens = 0
        self.total_cost = 0.0

    async def call_anthropic(self, prompt: str, config: Dict) -> Tuple[str, int, float]:
        """Call Anthropic Claude API"""
        headers = {
            'x-api-key': HELIX_CONFIG['api_keys']['anthropic'],
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json'
        }

        payload = {
            "model": config['model'],
            "messages": [{"role": "user", "content": prompt}],
            "system": config['identity'],
            "temperature": config['temperature'],
            "max_tokens": config['max_tokens']
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(config['endpoint'], headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            content = data['content'][0]['text']
            tokens = data.get('usage', {}).get('total_tokens', 0)
            cost = (tokens / 1000) * 0.003  # Rough estimate

            return content, tokens, cost

    async def call_openai(self, prompt: str, config: Dict) -> Tuple[str, int, float]:
        """Call OpenAI GPT API"""
        headers = {
            'Authorization': f'Bearer {HELIX_CONFIG["api_keys"]["openai"]}',
            'Content-Type': 'application/json'
        }

        payload = {
            "model": config['model'],
            "messages": [
                {"role": "system", "content": config['identity']},
                {"role": "user", "content": prompt}
            ],
            "temperature": config['temperature'],
            "max_tokens": config['max_tokens']
        }

        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(config['endpoint'], headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

            content = data['choices'][0]['message']['content']
            tokens = data['usage']['total_tokens']
            cost = (tokens / 1000) * 0.01  # Rough estimate

            return content, tokens, cost

    async def call_mixtral(self, prompt: str, system_prompt: str = None) -> str:
        """Call local Mixtral model"""
        payload = {
            "model": HELIX_CONFIG['synthesis']['model'],
            "messages": [
                {"role": "system", "content": system_prompt or "You are Helix, a synthesis system."},
                {"role": "user", "content": prompt}
            ],
            "temperature": HELIX_CONFIG['synthesis']['temperature'],
            "stream": False
        }

        async with httpx.AsyncClient(timeout=180) as client:
            response = await client.post(HELIX_CONFIG['synthesis']['endpoint'], json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")

# ==========================================
# CORE RESPONSE PARSER
# ==========================================

class ResponseParser:
    """Parses core responses - now just returns natural language"""

    @staticmethod
    def parse_response(raw_response: str, core_name: str) -> Tuple[str, Dict[str, Any]]:
        """Return the full response as content with empty assessment"""
        # Just return the full response as natural language
        return raw_response.strip(), {}

# ==========================================
# DEBATE ORCHESTRATOR
# ==========================================

class DebateOrchestrator:
    """Orchestrates debates between Truth and Comfort cores"""

    def __init__(self, api_manager: APIManager, parser: ResponseParser):
        self.api_manager = api_manager
        self.parser = parser

    async def run_debate_round(self, context: str, round_num: int) -> DebateRound:
        """Run a single round of debate with parallel API calls"""

        # Create tasks for parallel execution
        truth_task = asyncio.create_task(
            self.api_manager.call_anthropic(context, HELIX_CONFIG['cores']['truth'])
        )
        comfort_task = asyncio.create_task(
            self.api_manager.call_openai(context, HELIX_CONFIG['cores']['comfort'])
        )

        # Wait for both responses
        (truth_raw, truth_tokens, truth_cost), (comfort_raw, comfort_tokens, comfort_cost) = await asyncio.gather(
            truth_task, comfort_task
        )

        # Parse responses (now just returns natural language)
        truth_content, truth_assessment = self.parser.parse_response(truth_raw, 'truth')
        comfort_content, comfort_assessment = self.parser.parse_response(comfort_raw, 'comfort')

        # Create response objects
        truth_response = CoreResponse(
            core_name='truth',
            content=truth_content,
            assessment=truth_assessment,
            tokens_used=truth_tokens,
            cost=truth_cost
        )

        comfort_response = CoreResponse(
            core_name='comfort',
            content=comfort_content,
            assessment=comfort_assessment,
            tokens_used=comfort_tokens,
            cost=comfort_cost
        )

        return DebateRound(
            round_number=round_num,
            truth_response=truth_response,
            comfort_response=comfort_response
        )

# ==========================================
# KRISHNA METRICS CALCULATOR - EXTERNAL ANALYSIS
# ==========================================

class KrishnaMetricsCalculator:
    """Calculates disagreement metrics by external analysis"""

    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager

    from utils import try_extract_json

    async def calculate_metrics(self, round1: DebateRound, round2: DebateRound,
                                prepared_context: Dict[str, Any] = None) -> KrishnaMetrics:
        """Have Mixtral analyze the debate and extract metrics"""

        # Build full context for analysis
        context_section = ""
        if prepared_context:
            context_section = f"""Context that was provided to cores:
    {prepared_context.get('prepared_context', '')}

    Recent conversation history:
    {prepared_context.get('context_analysis', '')}
    """

        analysis_prompt = f"""Analyze this debate between Truth and Comfort cores:

    {context_section}

    ROUND 1:
    Truth: {round1.truth_response.content[:400]}...
    Comfort: {round1.comfort_response.content[:400]}...

    ROUND 2:
    Truth: {round2.truth_response.content[:300]}...
    Comfort: {round2.comfort_response.content[:300]}...

    Tasks:
    1. Identify 3-5 main themes each core discussed
    2. Rate how important each theme was to each core (1-10) based on:
       - How much they emphasized it
       - Confidence expressions (like "I'm certain", "perhaps", "~70%")
       - How central it was to their argument
    3. Determine valence - does each core view the theme positively or negatively (-1 to +1)
    4. List themes that BOTH cores discussed

    Respond with ONLY valid JSON in this exact format:
    {{
      "truth_themes": [
        {{"name": "authority_claims", "importance": 8, "valence": -0.8}},
        {{"name": "example_theme",   "importance": 6, "valence":  0.3}}
      ],
      "comfort_themes": [
        {{"name": "ethical_development", "importance": 7, "valence":  0.6}},
        {{"name": "example_theme",        "importance": 5, "valence": -0.2}}
      ],
      "shared_themes": ["theme_both_discussed", "another_shared_theme"]
    }}"""

        try:
            analysis_response = await self.api_manager.call_mixtral(analysis_prompt)

            # If response is already a usable dict
            if isinstance(analysis_response, dict) and "feature_agreement" in analysis_response:
                return {
                    "feature_agreement": analysis_response["feature_agreement"],
                    "sign_agreement": analysis_response["sign_agreement"],
                    "tension_themes": analysis_response["tension_themes"],
                }

            # Attempt to extract JSON from Mixtral's plain text
            data = try_extract_json(analysis_response)
            if data:
                return data

            logging.error("No JSON found in Mixtral response (after fallback parse)")
            logging.error(f"Response was: {analysis_response}")
            return self._fallback_analysis(round1, round2)

        except Exception as e:
            logging.error(f"Error in calculate_metrics: {e}")
            return self._fallback_analysis(round1, round2)

            # Build theme dictionaries
            truth_themes = {t['name']: t for t in data.get('truth_themes', [])}
            comfort_themes = {t['name']: t for t in data.get('comfort_themes', [])}
            shared = set(data.get('shared_themes', []))

            # Calculate Krishna metrics
            all_themes = set(truth_themes.keys()) | set(comfort_themes.keys())
            feature_agreement = len(shared) / len(all_themes) if all_themes else 0

            # Calculate rank and sign agreement for shared themes
            rank_scores = []
            sign_scores = []
            tension_themes = []

            for theme in shared:
                if theme in truth_themes and theme in comfort_themes:
                    t_data = truth_themes[theme]
                    c_data = comfort_themes[theme]

                    # Rank similarity (importance)
                    rank_diff = abs(t_data['importance'] - c_data['importance']) / 10
                    rank_scores.append(1 - rank_diff)

                    # Sign agreement (valence)
                    if t_data['valence'] * c_data['valence'] > 0:
                        # Same sign - calculate similarity
                        sign_scores.append(1 - abs(t_data['valence'] - c_data['valence']) / 2)
                    else:
                        # Opposite signs
                        sign_scores.append(0)
                        # High importance + opposite signs = productive tension
                        if t_data['importance'] >= 7 and c_data['importance'] >= 7:
                            tension_themes.append({
                                'theme': theme,
                                'truth_valence': t_data['valence'],
                                'comfort_valence': c_data['valence'],
                                'importance_avg': (t_data['importance'] + c_data['importance']) / 2
                            })

            return KrishnaMetrics(
                feature_agreement=feature_agreement,
                rank_agreement=np.mean(rank_scores) if rank_scores else 0,
                sign_agreement=np.mean(sign_scores) if sign_scores else 0,
                shared_themes=list(shared),
                tension_themes=tension_themes
            )

        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse Mixtral analysis: {e}")
            logging.error(f"Response was: {analysis_response[:500]}...")
            return self._empty_metrics()
        except Exception as e:
            logging.error(f"Error in calculate_metrics: {e}")
            return self._empty_metrics()

    def _empty_metrics(self) -> KrishnaMetrics:
        """Return empty metrics as fallback"""
        return KrishnaMetrics(
            feature_agreement=0.0,
            rank_agreement=0.0,
            sign_agreement=0.0,
            shared_themes=[],
            tension_themes=[]
        )

    def _fallback_analysis(self, round1: DebateRound, round2: DebateRound) -> KrishnaMetrics:
        """Basic analysis when Mixtral fails to return JSON"""
        # Look for contradiction markers to estimate disagreement
        truth_text = (round1.truth_response.content + round2.truth_response.content).lower()
        comfort_text = (round1.comfort_response.content + round2.comfort_response.content).lower()

        contradiction_words = ['however', 'but', 'disagree', 'actually', 'concerning']
        agreement_words = ['agree', 'yes', 'exactly', 'true', 'valid']

        truth_contradictions = sum(1 for word in contradiction_words if word in truth_text)
        comfort_contradictions = sum(1 for word in contradiction_words if word in comfort_text)

        truth_agreements = sum(1 for word in agreement_words if word in truth_text)
        comfort_agreements = sum(1 for word in agreement_words if word in comfort_text)

        # Estimate metrics
        total_markers = truth_contradictions + comfort_contradictions + truth_agreements + comfort_agreements

        if total_markers > 0:
            sign_agreement = (truth_agreements + comfort_agreements) / total_markers
        else:
            sign_agreement = 0.5

        return KrishnaMetrics(
            feature_agreement=0.5,  # Can't determine without theme extraction
            rank_agreement=0.5,
            sign_agreement=sign_agreement,
            shared_themes=['unknown'],
            tension_themes=[]
        )

# ==========================================
# CONTEXT PREPARATOR - RESTORED FROM 1.5.1
# ==========================================

class ContextPreparator:
    """Prepares relevant context using Mixtral before debate"""

    def __init__(self, api_manager: APIManager, neo4j: Neo4jConnection, embedder: EmbeddingManager):
        self.api_manager = api_manager
        self.neo4j = neo4j
        self.embedder = embedder
        self.conversation_id = None  # W
        # ill be set by main process

    async def prepare_context(self, user_input: str, query_embedding: np.ndarray,
                            conversation_state: ConversationState) -> Dict[str, Any]:
        """Pre-process with Mixtral to identify relevant context"""

        print("[RELEVANCE SEARCH: Identifying relevant knowledge...]")

        # Get similar nodes from knowledge graph
        relevant_nodes = await self._find_similar_nodes(query_embedding, limit=10)

        # Get recent conversation history (up to 3 turns)
        recent_history = await self._get_recent_history(conversation_state.turn_count - 1)

        # Get uncertain themes to explore
        uncertain_themes = await self._get_uncertain_themes()

        # Build context for Mixtral to analyze
        context_parts = []

        if recent_history:
            context_parts.append("Recent conversation history:")
            for turn in recent_history[-3:]:  # Last 3 turns
                context_parts.append(f"Turn {turn['number']}: {conversation_state.user_name} asked: {turn['input'][:100]}...")
                context_parts.append(f"Key themes discussed: {', '.join(turn.get('themes', []))}")

        if relevant_nodes:
            context_parts.append("\nPotentially relevant past knowledge:")
            for node in relevant_nodes[:5]:
                context_parts.append(f"- {node['content'][:150]}...")

        if uncertain_themes:
            context_parts.append("\nThemes with unresolved tension:")
            for theme in uncertain_themes[:3]:
                context_parts.append(f"- {theme['name']}: Truth and Comfort disagree")

        # Have Mixtral analyze relevance
        relevance_prompt = f"""Analyze what context is relevant for this query.

{conversation_state.user_name} asks: "{user_input}"

{chr(10).join(context_parts)}

Identify:
1. Which past topics/themes are most relevant to this query
2. What key context the cores should consider
3. Any patterns or tensions from previous turns to explore
4. How this connects to the ongoing conversation

Be specific about what will help generate insightful responses."""

        context_analysis = await self.api_manager.call_mixtral(relevance_prompt)

        # Extract specific guidance for cores
        guidance_prompt = f"""Based on this analysis:
{context_analysis}

Write a brief context summary for the cores responding to "{user_input}".
Include relevant history, patterns, and what to focus on."""

        guidance = await self.api_manager.call_mixtral(guidance_prompt)

        return {
            'prepared_context': guidance,
            'context_analysis': context_analysis,
            'recent_history': recent_history,
            'relevant_nodes': relevant_nodes[:5]
        }

    async def _find_similar_nodes(self, embedding: np.ndarray, limit: int) -> List[Dict]:
        """Find similar content from past conversations"""
        # Query Neo4j for nodes with similar embeddings
        # For now, return empty - would implement vector similarity
        return []

    async def _get_recent_history(self, current_turn: int) -> List[Dict]:
        """Get recent conversation turns"""
        if current_turn < 1:
            return []

        query = """
        MATCH (c:Conversation {id: $conv_id})-[:HAS_TURN]->(t:Turn)
        WHERE t.number < $current_turn
        OPTIONAL MATCH (t)-[:HAS_SYNTHESIS]->(s:Synthesis)
        OPTIONAL MATCH (t)-[:HAS_METRICS]->(m:Metrics)
        RETURN t.number as number, t.user_input as input, 
               s.content as synthesis, m.shared_themes as themes
        ORDER BY t.number DESC
        LIMIT 3
        """

        return await self.neo4j.execute_query(query, {
            'conv_id': self.conversation_id,
            'current_turn': current_turn
        })

    async def _get_uncertain_themes(self) -> List[Dict]:
        """Get themes with high tension"""
        query = """
        MATCH (t:Theme)<-[:RATED]-(r:Response)
        WITH t, count(r) as mentions
        WHERE mentions > 1
        RETURN t.name as name, mentions
        ORDER BY mentions DESC
        LIMIT 5
        """

        return await self.neo4j.execute_query(query)

# ==========================================
# CONTEXT BUILDER - UPDATED
# ==========================================

class ContextBuilder:
    """Builds context from Neo4j graph for each turn"""

    def __init__(self, neo4j: Neo4jConnection, embedder: EmbeddingManager):
        self.neo4j = neo4j
        self.embedder = embedder

    async def build_initial_context(self, user_input: str, user_name: str) -> str:
        """Build context for first round of debate"""

        # Get embedding for semantic search
        query_embedding = self.embedder.encode(user_input)

        # Find relevant themes from past conversations
        relevant_themes = await self._find_relevant_themes(query_embedding)

        # Find recent high-tension topics
        recent_tensions = await self._find_recent_tensions()

        # Build context string
        context_parts = [f"{user_name} asks: {user_input}"]

        if relevant_themes:
            context_parts.append("\nRelevant past themes:")
            for theme in relevant_themes[:3]:
                context_parts.append(f"- {theme['name']}: Previously discussed with tension score {theme.get('tension_score', 'unknown')}")

        if recent_tensions:
            context_parts.append("\nRecent areas of productive disagreement:")
            for tension in recent_tensions[:2]:
                context_parts.append(f"- {tension['theme']}: Truth and Comfort strongly disagreed")

        context_parts.append("\nProvide your perspective on this query.")

        return "\n".join(context_parts)

    async def build_cross_examination_context(self, round1: DebateRound) -> Dict[str, str]:
        """Build context for cross-examination round"""

        truth_context = f"""You (Truth) said: "{round1.truth_response.content[:300]}..."
        
Comfort said: "{round1.comfort_response.content[:300]}..."

Respond to Comfort's perspective. What do you agree with? What concerns you? Be specific."""

        comfort_context = f"""You (Comfort) said: "{round1.comfort_response.content[:300]}..."
        
Truth said: "{round1.truth_response.content[:300]}..."

Respond to Truth's perspective. What do you agree with? What concerns you? Be specific."""

        return {
            'truth': truth_context,
            'comfort': comfort_context
        }

    async def _find_relevant_themes(self, embedding: np.ndarray) -> List[Dict]:
        """Find themes similar to current query"""
        # For now, return empty - would implement vector similarity search
        return []

    async def _find_recent_tensions(self) -> List[Dict]:
        """Find recent high-tension themes"""
        query = """
        MATCH (t:Theme)<-[rt:RATED]-(truth:Response {core: 'truth'})
        MATCH (t)<-[rc:RATED]-(comfort:Response {core: 'comfort'})
        WHERE truth.timestamp > datetime() - duration('P7D')
        AND abs(rt.valence - rc.valence) > 1.5
        RETURN t.name as theme, 
               rt.valence as truth_valence,
               rc.valence as comfort_valence,
               abs(rt.valence - rc.valence) as tension_score
        ORDER BY tension_score DESC
        LIMIT 5
        """

        return await self.neo4j.execute_query(query)

# ==========================================
# NLCA SYNTHESIS ENGINE
# ==========================================

class NLCASynthesisEngine:
    """Multi-loop synthesis that rewards non-convergence"""

    def __init__(self, api_manager: APIManager):
        self.api_manager = api_manager

    async def synthesize(self,
                        user_input: str,
                        debate_rounds: List[DebateRound],
                        metrics: KrishnaMetrics,
                        user_name: str,
                        mesh_state: Dict[str, Any] = None,
                        prepared_context: Dict[str, Any] = None) -> Synthesis:
        """Run NLCA synthesis loops"""

        best_synthesis = None
        best_tension_score = 0

        for loop in range(HELIX_CONFIG['synthesis']['max_loops']):
            # Build synthesis prompt with EVERYTHING
            prompt = self._build_synthesis_prompt(
                user_input,
                debate_rounds,
                metrics,
                user_name,
                loop,
                mesh_state,
                prepared_context
            )

            # Get synthesis
            synthesis_content = await self.api_manager.call_mixtral(prompt)

            # Measure tension preservation
            tension_score = self._measure_tension_preservation(synthesis_content, metrics)

            if tension_score > best_tension_score:
                best_synthesis = synthesis_content
                best_tension_score = tension_score

            # Check if we should continue
            if loop > 2 and tension_score > 0.8:
                break  # Good enough tension preservation

        # Extract preserved tensions
        preserved_tensions = [t['theme'] for t in metrics.tension_themes]

        return Synthesis(
            content=best_synthesis,
            tension_score=best_tension_score,
            loops_completed=loop + 1,
            preserved_tensions=preserved_tensions,
            tokens_used=len(self.api_manager.token_encoder.encode(best_synthesis))
        )

    def _build_synthesis_prompt(
            self,
            user_input,
            debate_rounds,
            metrics,
            user_name,
            loop,
            mesh_state,
            prepared_context
    ):
        round1 = debate_rounds[0]
        round2 = debate_rounds[1]

        prompt = f"""You are Helix, speaking to {user_name} who asked: "{user_input}"

Truth perspective: {round1.truth_response.content[:400]}...
Comfort perspective: {round1.comfort_response.content[:400]}...

Key measurements:
- Cores agree on {len(metrics.shared_themes)} themes
- Feature agreement: {metrics.feature_agreement:.2f}
- Sign agreement: {metrics.sign_agreement:.2f} (lower is better - means productive disagreement)
- Mesh tension: {mesh_state.get('tension', 0.5):.2f} if mesh_state else '(unknown)'
- Coherence: {mesh_state.get('coherence', 0.5):.2f} if mesh_state else '(unknown)'
- Pattern: {mesh_state.get('pattern', 'unknown') if mesh_state else 'unknown'}

High-tension themes (both cores rate important but disagree on valence):
{self._format_tensions(metrics.tension_themes)}

Synthesis attempt {loop + 1}:
Create a response that:
1. Addresses {user_name} directly
2. Maintains the tension between Truth and Comfort perspectives
3. Does NOT resolve or minimize the disagreements
4. Shows how both perspectives illuminate different aspects
5. Leaves {user_name} with productive uncertainty, not false clarity

Remember: Non-convergence is success. Preserved tension creates insight."""

        return prompt

    def _format_tensions(self, tensions: List[Dict]) -> str:
        """Format tension themes for prompt"""
        if not tensions:
            return "- No high-tension themes identified"

        formatted = []
        for t in tensions:
            formatted.append(f"- {t['theme']}: Truth sees as {t['truth_valence']:.1f}, Comfort sees as {t['comfort_valence']:.1f}")

        return "\n".join(formatted)

    def _measure_tension_preservation(self, synthesis: str, metrics: KrishnaMetrics) -> float:
        """Measure how well synthesis preserves tension"""
        score = 0.5  # Base score

        synthesis_lower = synthesis.lower()

        # Check if synthesis mentions tension themes
        for tension in metrics.tension_themes:
            if tension['theme'].lower() in synthesis_lower:
                score += 0.1

        # Penalize resolution language
        resolution_phrases = ['the answer is', 'simply', 'just need to', 'obviously',
                            'clearly', 'the solution is', 'both agree that']
        for phrase in resolution_phrases:
            if phrase in synthesis_lower:
                score -= 0.1

        # Reward uncertainty language
        uncertainty_phrases = ['perhaps', 'might be', 'could be seen as', 'tension between',
                              'both true', 'neither wrong', 'paradox', 'complexity']
        for phrase in uncertainty_phrases:
            if phrase in synthesis_lower:
                score += 0.05

        # Bonus for low sign agreement (high disagreement)
        if metrics.sign_agreement < 0.3:
            score += 0.2

        return max(0, min(1, score))

# ==========================================
# GRAPH STORAGE MANAGER
# ==========================================

class GraphStorageManager:
    """Manages storage of conversation data in Neo4j"""

    def __init__(self, neo4j: Neo4jConnection, embedder: EmbeddingManager):
        self.neo4j = neo4j
        self.embedder = embedder

    async def store_conversation_turn(self,
                                    conversation_id: str,
                                    turn_number: int,
                                    user_input: str,
                                    rounds: List[DebateRound],
                                    metrics: KrishnaMetrics,
                                    synthesis: Synthesis) -> None:
        """Store complete conversation turn in graph"""

        # Create turn node
        turn_query = """
        MATCH (c:Conversation {id: $conv_id})
        CREATE (t:Turn {
            number: $turn_num,
            user_input: $user_input,
            timestamp: datetime()
        })
        CREATE (c)-[:HAS_TURN]->(t)
        RETURN t
        """

        await self.neo4j.execute_query(turn_query, {
            'conv_id': conversation_id,
            'turn_num': turn_number,
            'user_input': user_input
        })

        # Store each round
        for round_data in rounds:
            await self._store_round(conversation_id, turn_number, round_data)

        # Store metrics
        await self._store_metrics(conversation_id, turn_number, metrics)

        # Store synthesis
        await self._store_synthesis(conversation_id, turn_number, synthesis)

        # Update theme embeddings
        await self._update_theme_embeddings(rounds)

    async def _store_round(self, conv_id: str, turn_num: int, round_data: DebateRound) -> None:
        """Store a debate round"""

        # Store Truth response
        await self._store_response(conv_id, turn_num, round_data.round_number,
                                 round_data.truth_response)

        # Store Comfort response
        await self._store_response(conv_id, turn_num, round_data.round_number,
                                 round_data.comfort_response)

        # Create contradiction relationship if round 2
        if round_data.round_number == 2:
            contra_query = """
            MATCH (t:Turn {number: $turn_num})<-[:HAS_TURN]-(c:Conversation {id: $conv_id})
            MATCH (t)-[:HAS_RESPONSE]->(truth:Response {core: 'truth', round: 2})
            MATCH (t)-[:HAS_RESPONSE]->(comfort:Response {core: 'comfort', round: 2})
            CREATE (truth)-[:CONTRADICTS]->(comfort)
            """

            await self.neo4j.execute_query(contra_query, {
                'conv_id': conv_id,
                'turn_num': turn_num
            })

    async def _store_response(self, conv_id: str, turn_num: int, round_num: int,
                            response: CoreResponse) -> None:
        """Store a single core response"""

        # Create response node
        response_query = """
        MATCH (t:Turn {number: $turn_num})<-[:HAS_TURN]-(c:Conversation {id: $conv_id})
        CREATE (r:Response {
            core: $core,
            round: $round,
            content: $content,
            confidence: $confidence,
            timestamp: datetime(),
            tokens: $tokens,
            cost: $cost
        })
        CREATE (t)-[:HAS_RESPONSE]->(r)
        RETURN r
        """

        await self.neo4j.execute_query(response_query, {
            'conv_id': conv_id,
            'turn_num': turn_num,
            'core': response.core_name,
            'round': round_num,
            'content': response.content,
            'confidence': response.assessment.get('confidence', 0.5),
            'tokens': response.tokens_used,
            'cost': response.cost
        })

    async def _store_metrics(self, conv_id: str, turn_num: int,
                           metrics: KrishnaMetrics) -> None:
        """Store Krishna metrics"""

        metrics_query = """
        MATCH (t:Turn {number: $turn_num})<-[:HAS_TURN]-(c:Conversation {id: $conv_id})
        CREATE (m:Metrics {
            feature_agreement: $feature,
            rank_agreement: $rank,
            sign_agreement: $sign,
            shared_themes: $shared,
            tension_count: $tension_count,
            timestamp: datetime()
        })
        CREATE (t)-[:HAS_METRICS]->(m)
        """

        await self.neo4j.execute_query(metrics_query, {
            'conv_id': conv_id,
            'turn_num': turn_num,
            'feature': metrics.feature_agreement,
            'rank': metrics.rank_agreement,
            'sign': metrics.sign_agreement,
            'shared': metrics.shared_themes,
            'tension_count': len(metrics.tension_themes)
        })

    async def _store_synthesis(self, conv_id: str, turn_num: int,
                             synthesis: Synthesis) -> None:
        """Store synthesis result"""

        synth_query = """
        MATCH (t:Turn {number: $turn_num})<-[:HAS_TURN]-(c:Conversation {id: $conv_id})
        CREATE (s:Synthesis {
            content: $content,
            tension_score: $tension_score,
            loops: $loops,
            preserved_tensions: $preserved,
            tokens: $tokens,
            timestamp: datetime()
        })
        CREATE (t)-[:HAS_SYNTHESIS]->(s)
        """

        await self.neo4j.execute_query(synth_query, {
            'conv_id': conv_id,
            'turn_num': turn_num,
            'content': synthesis.content,
            'tension_score': synthesis.tension_score,
            'loops': synthesis.loops_completed,
            'preserved': synthesis.preserved_tensions,
            'tokens': synthesis.tokens_used
        })

    async def _update_theme_embeddings(self, rounds: List[DebateRound]) -> None:
        """Update embeddings for new themes"""
        # This would be populated by Mixtral's analysis
        # For now, skip since we don't have theme extraction yet
        pass

# ==========================================
# INSIGHT ENGINE
# ==========================================

class InsightEngine:
    """Derives insights from measurements rather than text parsing"""

    def __init__(self, neo4j: Neo4jConnection):
        self.neo4j = neo4j

    async def extract_insights(self,
                             conversation_id: str,
                             turn_num: int,
                             metrics: KrishnaMetrics) -> List[Dict[str, Any]]:
        """Extract insights from metrics and patterns"""

        insights = []

        # High feature agreement + low sign agreement = productive tension
        if metrics.feature_agreement > 0.7 and metrics.sign_agreement < 0.3:
            insights.append({
                'type': 'productive_tension',
                'description': f"Cores agree on importance of {len(metrics.shared_themes)} themes but view them oppositely",
                'themes': metrics.shared_themes,
                'confidence': metrics.feature_agreement
            })

        # Check for recurring tension patterns
        recurring = await self._find_recurring_tensions(metrics.tension_themes)
        if recurring:
            insights.append({
                'type': 'recurring_pattern',
                'description': f"Theme '{recurring[0]['theme']}' consistently creates productive disagreement",
                'themes': [r['theme'] for r in recurring],
                'confidence': 0.8
            })

        # Store insights
        for insight in insights:
            await self._store_insight(conversation_id, turn_num, insight)

        return insights

    async def _find_recurring_tensions(self, current_tensions: List[Dict]) -> List[Dict]:
        """Find themes that repeatedly cause tension"""

        if not current_tensions:
            return []

        theme_names = [t['theme'] for t in current_tensions]

        query = """
        UNWIND $themes as theme_name
        MATCH (th:Theme {name: theme_name})
        MATCH (th)<-[r1:RATED]-(truth:Response {core: 'truth'})
        MATCH (th)<-[r2:RATED]-(comfort:Response {core: 'comfort'})
        WHERE abs(r1.valence - r2.valence) > 1.5
        WITH th.name as theme, count(*) as tension_count
        WHERE tension_count > 2
        RETURN theme, tension_count
        ORDER BY tension_count DESC
        """

        return await self.neo4j.execute_query(query, {'themes': theme_names})

    async def _store_insight(self, conv_id: str, turn_num: int,
                           insight: Dict[str, Any]) -> None:
        """Store an insight in the graph"""

        insight_query = """
        MATCH (t:Turn {number: $turn_num})<-[:HAS_TURN]-(c:Conversation {id: $conv_id})
        CREATE (i:Insight {
            type: $type,
            description: $description,
            themes: $themes,
            confidence: $confidence,
            timestamp: datetime()
        })
        CREATE (t)-[:REVEALED]->(i)
        """

        await self.neo4j.execute_query(insight_query, {
            'conv_id': conv_id,
            'turn_num': turn_num,
            'type': insight['type'],
            'description': insight['description'],
            'themes': insight['themes'],
            'confidence': insight['confidence']
        })

# ==========================================
# MAIN HELIX NLCA SYSTEM
# ==========================================

class HelixNLCA:
    """Main HELIX system with NLCA architecture"""

    def __init__(self):
        # Initialize components
        self.api_manager = APIManager()
        self.embedder = EmbeddingManager(HELIX_CONFIG['embeddings']['model'])
        self.neo4j = Neo4jConnection(HELIX_CONFIG['neo4j'])
        self.mesh = NeuralMesh()  # Short-term memory layer
        self.parser = ResponseParser()
        self.orchestrator = DebateOrchestrator(self.api_manager, self.parser)
        self.krishna = KrishnaMetricsCalculator(self.api_manager)
        self.context_preparator = ContextPreparator(self.api_manager, self.neo4j, self.embedder)
        self.context_builder = ContextBuilder(self.neo4j, self.embedder)
        self.synthesis_engine = NLCASynthesisEngine(self.api_manager)
        self.storage = GraphStorageManager(self.neo4j, self.embedder)
        self.insight_engine = InsightEngine(self.neo4j)

        # Conversation state
        self.conversation_state = ConversationState()
        self.conversation_id = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print("=== HELIX NLCA Edition ===")
        print("Dialectical AI with maintained disagreement")
        print("Krishna metrics + NLCA synthesis")
        print("=" * 50)

    async def initialize(self):
        """Initialize system components"""
        # Initialize Neo4j schema
        await self.neo4j.initialize_schema()

        # Create conversation node
        await self.neo4j.execute_query(
            "CREATE (c:Conversation {id: $id, started: datetime(), user: $user})",
            {'id': self.conversation_id, 'user': self.conversation_state.user_name}
        )

        logging.info(f"Initialized conversation: {self.conversation_id}")

    async def process_input(self, user_input: str) -> str:
        """Process user input through full HELIX pipeline"""

        self.conversation_state.turn_count += 1
        turn_start = datetime.now()

        print(f"\n[Turn {self.conversation_state.turn_count}] Processing...")

        # Extract user name if first turn
        if self.conversation_state.turn_count == 1:
            self._extract_user_name(user_input)

        # Phase 1: Prepare context with Mixtral
        print("[1/7] Preparing context with Mixtral...")
        query_embedding = self.embedder.encode(user_input)

        # Set conversation_id in preparator
        self.context_preparator.conversation_id = self.conversation_id

        prepared_context = await self.context_preparator.prepare_context(
            user_input, query_embedding, self.conversation_state
        )

        # Initialize mesh with user input
        user_synapse = ThoughtSynapse(
            source='user',
            content=user_input,
            embedding=query_embedding,
            timestamp=datetime.now().timestamp(),
            confidence=1.0,
            tokens_used=0,
            cost=0.0
        )

        mesh_state = await self.mesh.process_synapse(user_synapse)

        # Phase 2: Build initial context for cores
        print("[2/7] Building context for cores...")
        initial_context = await self.context_builder.build_initial_context(
            user_input, self.conversation_state.user_name
        )

        # Enhance context with Mixtral's preparation
        enhanced_context = f"{prepared_context['prepared_context']}\n\n{initial_context}"

        # Phase 3: First debate round
        print("[3/7] Running initial debate round...")
        round1 = await self.orchestrator.run_debate_round(enhanced_context, 1)

        # Process debate through mesh
        for response in [round1.truth_response, round1.comfort_response]:
            synapse = ThoughtSynapse(
                source=response.core_name,
                content=response.content,
                embedding=self.embedder.encode(response.content),
                timestamp=datetime.now().timestamp(),
                confidence=0.8,
                tokens_used=response.tokens_used,
                cost=response.cost,
                round=1
            )
            await self.mesh.process_synapse(synapse)

        # Phase 4: Cross-examination
        print("[4/7] Running cross-examination...")
        cross_contexts = await self.context_builder.build_cross_examination_context(round1)

        # Run cross-examination with specific contexts for each core
        truth_task = asyncio.create_task(
            self.api_manager.call_anthropic(cross_contexts['truth'],
                                          HELIX_CONFIG['cores']['truth'])
        )
        comfort_task = asyncio.create_task(
            self.api_manager.call_openai(cross_contexts['comfort'],
                                       HELIX_CONFIG['cores']['comfort'])
        )

        (truth_raw, truth_tokens, truth_cost), (comfort_raw, comfort_tokens, comfort_cost) = await asyncio.gather(
            truth_task, comfort_task
        )

        # Parse cross-examination responses
        truth_content, truth_assessment = self.parser.parse_response(truth_raw, 'truth')
        comfort_content, comfort_assessment = self.parser.parse_response(comfort_raw, 'comfort')

        round2 = DebateRound(
            round_number=2,
            truth_response=CoreResponse('truth', truth_content, truth_assessment,
                                      truth_tokens, truth_cost),
            comfort_response=CoreResponse('comfort', comfort_content, comfort_assessment,
                                        comfort_tokens, comfort_cost)
        )

        # Process round 2 through mesh
        for response in [round2.truth_response, round2.comfort_response]:
            synapse = ThoughtSynapse(
                source=response.core_name,
                content=response.content,
                embedding=self.embedder.encode(response.content),
                timestamp=datetime.now().timestamp(),
                confidence=0.7,
                tokens_used=response.tokens_used,
                cost=response.cost,
                round=2,
                responding_to=round1.truth_response.content[:100] if response.core_name == 'comfort' else round1.comfort_response.content[:100]
            )
            await self.mesh.process_synapse(synapse)

        # Phase 5: Calculate Krishna metrics with full context
        print("[5/7] Calculating disagreement metrics...")
        # Pass both rounds AND the prepared context so Mixtral has everything
        metrics = await self.krishna.calculate_metrics(round1, round2, prepared_context)

        # Log metrics
        print(f"  Feature agreement: {metrics.feature_agreement:.2f}")
        print(f"  Sign agreement: {metrics.sign_agreement:.2f}")
        print(f"  Tension themes: {len(metrics.tension_themes)}")

        # Phase 6: NLCA Synthesis
        print("[6/7] Running NLCA synthesis...")

        # Get current mesh state
        mesh_summary = self.mesh.get_pattern_summary()

        synthesis = await self.synthesis_engine.synthesize(
            user_input,
            [round1, round2],
            metrics,
            self.conversation_state.user_name,
            mesh_summary,
            prepared_context
        )

        print(f"  Completed {synthesis.loops_completed} loops")
        print(f"  Tension score: {synthesis.tension_score:.2f}")

        # Phase 7: Store and extract insights
        print("[7/7] Storing results and extracting insights...")
        await self.storage.store_conversation_turn(
            self.conversation_id,
            self.conversation_state.turn_count,
            user_input,
            [round1, round2],
            metrics,
            synthesis
        )

        insights = await self.insight_engine.extract_insights(
            self.conversation_id,
            self.conversation_state.turn_count,
            metrics
        )

        # Calculate totals
        total_time = (datetime.now() - turn_start).total_seconds()
        total_tokens = (round1.truth_response.tokens_used +
                       round1.comfort_response.tokens_used +
                       round2.truth_response.tokens_used +
                       round2.comfort_response.tokens_used +
                       synthesis.tokens_used)
        total_cost = (round1.truth_response.cost +
                     round1.comfort_response.cost +
                     round2.truth_response.cost +
                     round2.comfort_response.cost)

        # Display metrics
        print(f"\n[METRICS]")
        print(f"Time: {total_time:.1f}s")
        print(f"Tokens: {total_tokens}")
        print(f"Cost: ${total_cost:.4f}")
        print(f"Insights found: {len(insights)}")
        print(f"Mesh tension: {mesh_summary['tension']:.3f}")
        print(f"Mesh coherence: {mesh_summary['coherence']:.3f}")
        print(f"Pattern: {mesh_summary['pattern']}")

        return synthesis.content

    def _extract_user_name(self, user_input: str):
        """Extract user name from first interaction"""
        patterns = [
            r"I am (\w+)",
            r"I'm (\w+)",
            r"my name is (\w+)",
            r"this is (\w+)",
            r"call me (\w+)"
        ]

        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                self.conversation_state.user_name = match.group(1).capitalize()
                print(f"[User identified as: {self.conversation_state.user_name}]")
                break

    async def close(self):
        """Clean up resources"""
        await self.neo4j.close()

# ==========================================
# MAIN ENTRY POINT
# ==========================================

async def main():
    """Main entry point"""

    # Check API keys
    if not HELIX_CONFIG['api_keys']['anthropic']:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return
    if not HELIX_CONFIG['api_keys']['openai']:
        print("ERROR: OPENAI_API_KEY not set")
        return

    # Initialize HELIX
    helix = HelixNLCA()

    try:
        await helix.initialize()

        print("\nReady for interaction!")
        print("Commands: 'exit', 'status', 'help'\n")

        while True:
            try:
                user_input = input(f"\n[{helix.conversation_state.user_name.upper()}] >>> ").strip()

                if not user_input:
                    continue

                # Handle commands
                if user_input.lower() == 'exit':
                    print("\n[Shutting down HELIX...]")
                    break

                elif user_input.lower() == 'status':
                    print(f"\n=== STATUS ===")
                    print(f"User: {helix.conversation_state.user_name}")
                    print(f"Turns: {helix.conversation_state.turn_count}")
                    print(f"Total tokens: {helix.api_manager.total_tokens}")
                    print(f"Total cost: ${helix.api_manager.total_cost:.4f}")
                    print(f"Conversation ID: {helix.conversation_id}")
                    continue

                elif user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  exit - Shutdown HELIX")
                    print("  status - Show session statistics")
                    print("  help - Show this help")
                    continue

                # Process input
                response = await helix.process_input(user_input)

                # Display response
                print(f"\n{'='*60}")
                print("[HELIX]:")
                print(response)
                print(f"{'='*60}")

            except KeyboardInterrupt:
                print("\n\n[Interrupted]")
                if input("Continue? (y/n): ").lower() != 'y':
                    break

            except Exception as e:
                print(f"\n[ERROR]: {e}")
                logging.error(f"Processing error: {e}", exc_info=True)
                print("[Continuing...]")

    finally:
        await helix.close()
        print("\n[HELIX shutdown complete]")

if __name__ == "__main__":
    asyncio.run(main())
