import os
import json
import hashlib
import time
import requests
from typing import List, Dict, Optional, AsyncGenerator, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from sentence_transformers import SentenceTransformer, CrossEncoder
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text
import redis.asyncio as redis
import numpy as np

class ChatHandler:
    def __init__(self, engine: Optional = None):
        self.engine = engine
        self.ollama_url = "http://ollama:11434/api/generate"
        self.model = "mistral"
        
        print("🚀 Initialisation ChatHandler ULTRA-PRO + Multi-Modal...")
        
        # Redis Cache
        try:
            self.redis = redis.from_url("redis://redis:6379/0", decode_responses=False)
            self.cache_enabled = True
            print("✅ Redis Cache active")
        except Exception as e:
            print(f"⚠️ Redis non disponible: {e}")
            self.cache_enabled = False
        
        # Embedding Model
        try:
            self.embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
            print("✅ Embedding model charge")
        except Exception as e:
            print(f"❌ Erreur embedding model: {e}")
            self.embedding_model = None
        
        # Reranker Model
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✅ Reranker charge")
        except Exception as e:
            print(f"⚠️ Reranker non disponible: {e}")
            self.reranker = None
        
        # Qdrant
        try:
            self.qdrant_client = QdrantClient(host="qdrant", port=6333)
            self.collection_name = "legal_folders"
            info = self.qdrant_client.get_collection(collection_name=self.collection_name)
            print(f"✅ Qdrant OK ({info.points_count} vecteurs)")
            self.use_rag = True
        except Exception as e:
            print(f"❌ Qdrant non disponible: {e}")
            self.use_rag = False
        
        # Ollama
        try:
            response = requests.get("http://ollama:11434/api/tags", timeout=2)
            self.use_ollama = response.status_code == 200
            if self.use_ollama:
                print(f"✅ Ollama OK (modele: {self.model})")
        except:
            self.use_ollama = False
            print("⚠️ Ollama non disponible")
        
        # Multi-Modal
        try:
            from app.multimodal import MultiModalProcessor
            self.multimodal = MultiModalProcessor()
            print("✅ Multi-Modal processor integre")
        except Exception as e:
            print(f"⚠️ Multi-Modal non disponible: {e}")
            self.multimodal = None
        
        # Synonymes juridiques
        self.legal_synonyms = {
            'bail': ['bail', 'location', 'contrat de location', 'loyer'],
            'bailleur': ['bailleur', 'propriétaire', 'loueur', 'propriétaire bailleur'],
            'locataire': ['locataire', 'preneur', 'preneur à bail', 'occupant'],
            'plainte': ['plainte', 'dépôt de plainte', 'réclamation', 'grief'],
            'fraude': ['fraude', 'escroquerie', 'tromperie', 'malversation'],
            'contrat': ['contrat', 'convention', 'accord', 'engagement'],
            'obligation': ['obligation', 'devoir', 'responsabilité', 'engagement'],
            'recours': ['recours', 'action en justice', 'voie de recours', 'procédure'],
            'résiliation': ['résiliation', 'rupture', 'fin de contrat', 'cessation'],
            'indemnité': ['indemnité', 'dédommagement', 'réparation', 'compensation']
        }
        
        # Metrics
        self.metrics = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': [],
            'query_types': Counter(),
            'top_queries': Counter()
        }
        
        # Auto-tuning weights
        self.weights = {
            'vector': 0.7,
            'fulltext': 0.3,
            'rerank_boost': 1.2
        }
        
        print(f"\n🎯 ChatHandler COMPLETE pret !")
        print(f"   - RAG: {self.use_rag}")
        print(f"   - Ollama: {self.use_ollama}")
        print(f"   - Cache: {self.cache_enabled}")
        print(f"   - Reranker: {self.reranker is not None}")
        print(f"   - Multi-Modal: {self.multimodal is not None}")
        print(f"   - Query Expansion: {len(self.legal_synonyms)} termes")
        print(f"   - Metrics: Activé")
        print(f"   - Auto-tuning: Activé\n")

    async def process_message(self, user_message: str, folders: List[Dict], 
                             filters: Optional[Dict] = None) -> str:
        """Point d'entrée principal"""
        
        start_time = time.time()
        self.metrics['total_queries'] += 1
        
        # Cache check
        if self.cache_enabled:
            cache_key = self._generate_cache_key(user_message, filters)
            cached = await self._get_from_cache(cache_key)
            if cached:
                self.metrics['cache_hits'] += 1
                elapsed = time.time() - start_time
                self.metrics['avg_response_time'].append(elapsed)
                print(f"💨 Cache HIT ({elapsed*1000:.0f}ms)")
                return cached + "\n\n[⚡ Réponse depuis cache]"
            else:
                self.metrics['cache_misses'] += 1
        
        # Processing
        if self.use_rag and self.use_ollama:
            response = await self._process_with_rag_ultra_pro(user_message, filters)
        elif self.use_rag:
            response = self._process_rag_only(user_message, filters)
        else:
            response = self._process_basic(user_message, folders)
        
        # Cache store
        if self.cache_enabled and response:
            await self._store_in_cache(cache_key, response, ttl=3600)
        
        # Metrics
        elapsed = time.time() - start_time
        self.metrics['avg_response_time'].append(elapsed)
        self.metrics['top_queries'][user_message[:50]] += 1
        
        print(f"⏱️ Response time: {elapsed*1000:.0f}ms")
        
        return response

    async def process_message_stream(self, user_message: str, folders: List[Dict],
                                    filters: Optional[Dict] = None) -> AsyncGenerator[str, None]:
        """Streaming version"""
        
        start_time = time.time()
        self.metrics['total_queries'] += 1
        
        # Cache check
        if self.cache_enabled:
            cache_key = self._generate_cache_key(user_message, filters)
            cached = await self._get_from_cache(cache_key)
            if cached:
                self.metrics['cache_hits'] += 1
                yield cached
                yield "\n\n[⚡ Réponse depuis cache]"
                return
            else:
                self.metrics['cache_misses'] += 1
        
        # LLM Multi-query
        yield "[🧠 Analyse de la requête...]\n"
        sub_queries = await self._llm_decompose_query(user_message)
        yield f"[🔍 Décomposée en {len(sub_queries)} sous-requêtes]\n\n"
        
        # Query expansion
        expanded_queries = []
        for sq in sub_queries:
            expanded = self._expand_query(sq)
            expanded_queries.extend(expanded)
        
        yield f"[📚 Expansion: {len(expanded_queries)} requêtes totales]\n"
        
        # Hybrid search with cache
        all_docs = []
        for eq in expanded_queries[:10]:
            docs = await self._hybrid_search_with_cache(eq, top_k=5, filters=filters)
            all_docs.extend(docs)
        
        # Deduplication
        seen = set()
        unique_docs = []
        for doc in all_docs:
            if doc['id'] not in seen:
                seen.add(doc['id'])
                unique_docs.append(doc)
        
        yield f"[🎯 {len(unique_docs)} documents uniques trouvés]\n"
        
        # Reranking
        if self.reranker and len(unique_docs) > 3:
            unique_docs = self._rerank_results_boosted(user_message, unique_docs)
            yield "[🔄 Reranking appliqué]\n\n"
        
        top_docs = unique_docs[:10]
        
        if not top_docs:
            yield "Aucun dossier pertinent trouvé."
            return
        
        # Context
        context = "\n".join([f"{i+1}. {doc['name']} (score: {doc['score']:.3f})"
                            for i, doc in enumerate(top_docs)])
        
        prompt = f"""Tu es un assistant juridique expert pour LegalMind.

DOSSIERS PERTINENTS (triés par pertinence):
{context}

QUESTION UTILISATEUR: {user_message}

INSTRUCTIONS:
- Réponds de manière professionnelle, précise et structurée
- Cite explicitement les dossiers pertinents avec [Dossier N]
- Structure ta réponse avec des paragraphes clairs
- Sois concis mais exhaustif (max 600 mots)

RÉPONSE:"""
        
        # Streaming Ollama
        full_response = ""
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 1000,
                        "top_k": 40,
                        "top_p": 0.9
                    }
                },
                stream=True,
                timeout=90
            )
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            full_response += chunk
                            yield chunk
                    except json.JSONDecodeError:
                        continue
            
            # Sources
            sources = self._format_sources_enhanced(top_docs[:5])
            yield f"\n\n{sources}"
            
            # Metrics footer
            elapsed = time.time() - start_time
            cache_rate = (self.metrics['cache_hits'] / max(1, self.metrics['total_queries'])) * 100
            yield f"\n\n---\n⏱️ {elapsed:.1f}s | 📊 Cache: {cache_rate:.0f}% | 🎯 {len(top_docs)} docs"
            
            # Cache
            if self.cache_enabled and full_response:
                await self._store_in_cache(cache_key, full_response + "\n\n" + sources, ttl=3600)
            
            # Update metrics
            self.metrics['avg_response_time'].append(elapsed)
            self.metrics['top_queries'][user_message[:50]] += 1
            
        except Exception as e:
            yield f"\n\n⚠️ Erreur: {str(e)}"

    async def _llm_decompose_query(self, query: str) -> List[str]:
        """Multi-Query avec LLM"""
        
        if not self.use_ollama:
            return self._decompose_query_rules(query)
        
        try:
            prompt = f"""Tu es un expert en décomposition de requêtes juridiques.

REQUÊTE: {query}

TÂCHE: Décompose cette requête en 2-4 sous-requêtes simples pour améliorer la recherche.

RÈGLES:
- Chaque sous-requête autonome
- Couvrir tous les aspects
- Pas de redondances
- Format: une par ligne
- Pas de numérotation

SOUS-REQUÊTES:"""

            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 200}
                },
                timeout=15
            )
            
            if response.status_code == 200:
                answer = response.json().get("response", "")
                sub_queries = [q.strip() for q in answer.split('\n') if q.strip() and len(q.strip()) > 10]
                sub_queries.insert(0, query)
                return sub_queries[:5]
            
        except Exception as e:
            print(f"⚠️ LLM decomposition failed: {e}")
        
        return self._decompose_query_rules(query)

    def _decompose_query_rules(self, query: str) -> List[str]:
        """Fallback règles"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['et', 'ou', 'ainsi que']):
            sub_queries = [query]
            
            if 'bailleur' in query_lower and 'locataire' in query_lower:
                sub_queries.append(query_lower.replace('et', '').replace('locataire', '').strip())
                sub_queries.append(query_lower.replace('et', '').replace('bailleur', '').strip())
            
            if 'obligation' in query_lower and 'recours' in query_lower:
                parts = query_lower.split('et')
                sub_queries.extend([p.strip() for p in parts if len(p.strip()) > 10])
            
            return list(set([q for q in sub_queries if len(q) > 10]))[:5]
        
        return [query]

    def _expand_query(self, query: str) -> List[str]:
        """Query Expansion"""
        expanded = [query]
        query_lower = query.lower()
        
        for term, synonyms in self.legal_synonyms.items():
            if term in query_lower:
                for synonym in synonyms[:2]:
                    if synonym != term:
                        expanded_query = query_lower.replace(term, synonym)
                        if expanded_query not in expanded:
                            expanded.append(expanded_query)
        
        return expanded[:3]

    async def _hybrid_search_with_cache(self, query: str, top_k=10, 
                                       filters: Optional[Dict] = None) -> List[Dict]:
        """Hybrid Search avec cache embeddings"""
        
        # Check embedding cache
        embedding_cache_key = f"emb:{hashlib.md5(query.encode()).hexdigest()}"
        query_vector = None
        
        if self.cache_enabled:
            cached_embedding = await self._get_embedding_from_cache(embedding_cache_key)
            if cached_embedding:
                query_vector = cached_embedding
        
        # Calculate embedding
        if query_vector is None and self.embedding_model:
            query_vector = self.embedding_model.encode(query).tolist()
            if self.cache_enabled:
                await self._store_embedding_in_cache(embedding_cache_key, query_vector, ttl=86400)
        
        results = []
        
        # Vector Search
        if self.use_rag and query_vector:
            vector_results = self._vector_search_direct(query_vector, top_k=top_k, filters=filters)
            for doc in vector_results:
                doc['search_type'] = 'vector'
                doc['score'] = doc['score'] * self.weights['vector']
            results.extend(vector_results)
        
        # Full-Text Search
        if self.engine:
            try:
                with self.engine.connect() as conn:
                    sql = text("""
                        SELECT id, name, 
                               ts_rank(to_tsvector('french', name), plainto_tsquery('french', :query)) as score
                        FROM folders
                        WHERE to_tsvector('french', name) @@ plainto_tsquery('french', :query)
                        ORDER BY score DESC
                        LIMIT :limit
                    """)
                    fulltext_results = conn.execute(sql, {"query": query, "limit": top_k}).fetchall()
                    
                    for row in fulltext_results:
                        results.append({
                            'id': row[0],
                            'name': row[1],
                            'score': float(row[2]) * self.weights['fulltext'],
                            'search_type': 'fulltext'
                        })
            except Exception as e:
                print(f"⚠️ Full-text error: {e}")
        
        # Merge
        merged = {}
        for doc in results:
            if doc['id'] in merged:
                merged[doc['id']]['score'] += doc['score']
                merged[doc['id']]['search_type'] = 'hybrid'
            else:
                merged[doc['id']] = doc
        
        return sorted(merged.values(), key=lambda x: x['score'], reverse=True)[:top_k]

    def _vector_search_direct(self, query_vector: List[float], top_k=10, 
                             filters: Optional[Dict] = None) -> List[Dict]:
        """Vector search direct"""
        if not self.use_rag:
            return []
        
        try:
            qdrant_filter = None
            if filters:
                conditions = []
                if 'date_from' in filters:
                    conditions.append({"key": "created_at", "range": {"gte": filters['date_from']}})
                if 'date_to' in filters:
                    conditions.append({"key": "created_at", "range": {"lte": filters['date_to']}})
                if 'category' in filters:
                    conditions.append({"key": "category", "match": {"value": filters['category']}})
                
                if conditions:
                    qdrant_filter = {"must": conditions}
            
            results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=0.3,
                query_filter=qdrant_filter
            )
            
            return [{
                'id': hit.id,
                'name': hit.payload.get('name', ''),
                'score': hit.score,
                'metadata': hit.payload
            } for hit in results.points]
            
        except Exception as e:
            print(f"❌ Vector search error: {e}")
            return []

    def _rerank_results_boosted(self, query: str, docs: List[Dict]) -> List[Dict]:
        """Rerank avec boost"""
        if not self.reranker or not docs:
            return docs
        
        try:
            pairs = [(query, doc['name']) for doc in docs]
            scores = self.reranker.predict(pairs)
            
            for doc, score in zip(docs, scores):
                doc['rerank_score'] = float(score)
                doc['score'] = (doc['score'] + float(score) * self.weights['rerank_boost']) / 2
            
            self._update_weights(docs)
            
            return sorted(docs, key=lambda x: x['score'], reverse=True)
        except Exception as e:
            print(f"⚠️ Reranking error: {e}")
            return docs

    def _update_weights(self, docs: List[Dict]):
        """Auto-tuning"""
        if len(docs) < 5:
            return
        
        vector_scores = [d['score'] for d in docs if d.get('search_type') == 'vector']
        fulltext_scores = [d['score'] for d in docs if d.get('search_type') == 'fulltext']
        
        if vector_scores and fulltext_scores:
            avg_vector = np.mean(vector_scores)
            avg_fulltext = np.mean(fulltext_scores)
            
            total = avg_vector + avg_fulltext
            if total > 0:
                new_vector_weight = (avg_vector / total) * 0.05 + self.weights['vector'] * 0.95
                new_fulltext_weight = (avg_fulltext / total) * 0.05 + self.weights['fulltext'] * 0.95
                
                total_weight = new_vector_weight + new_fulltext_weight
                self.weights['vector'] = new_vector_weight / total_weight
                self.weights['fulltext'] = new_fulltext_weight / total_weight

    async def _process_with_rag_ultra_pro(self, user_message: str, 
                                         filters: Optional[Dict] = None) -> str:
        """Version non-streaming"""
        try:
            sub_queries = await self._llm_decompose_query(user_message)
            
            expanded_queries = []
            for sq in sub_queries:
                expanded = self._expand_query(sq)
                expanded_queries.extend(expanded)
            
            all_docs = []
            for eq in expanded_queries[:10]:
                docs = await self._hybrid_search_with_cache(eq, top_k=5, filters=filters)
                all_docs.extend(docs)
            
            seen = set()
            unique_docs = []
            for doc in all_docs:
                if doc['id'] not in seen:
                    seen.add(doc['id'])
                    unique_docs.append(doc)
            
            if self.reranker and len(unique_docs) > 3:
                unique_docs = self._rerank_results_boosted(user_message, unique_docs)
            
            relevant_docs = unique_docs[:10]
            
            if not relevant_docs:
                return "Aucun dossier pertinent trouvé."
            
            context = "\n".join([f"{i+1}. {doc['name']} (score: {doc['score']:.3f})"
                                for i, doc in enumerate(relevant_docs)])
            
            prompt = f"""Tu es un assistant juridique expert.

DOSSIERS:
{context}

QUESTION: {user_message}

RÉPONSE:"""
            
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "num_predict": 1000}
                },
                timeout=90
            )
            
            if response.status_code == 200:
                answer = response.json().get("response", "")
                sources = self._format_sources_enhanced(relevant_docs[:5])
                return f"{answer}\n\n{sources}"
            else:
                return self._process_rag_only(user_message, filters)
                
        except Exception as e:
            return f"⚠️ Erreur: {str(e)}"

    def _process_rag_only(self, user_message: str, filters: Optional[Dict] = None) -> str:
        """RAG sans Ollama"""
        relevant_docs = self._vector_search_legacy(user_message, top_k=15, filters=filters)
        
        if not relevant_docs:
            return "Aucun dossier correspondant."
        
        categories = defaultdict(list)
        for doc in relevant_docs:
            name = doc['name'].lower()
            if 'plainte' in name:
                categories['📋 Plaintes'].append(doc)
            elif 'argumentaire' in name:
                categories['📝 Argumentaires'].append(doc)
            elif 'vulnérabilité' in name:
                categories['⚠️ Vulnérabilité'].append(doc)
            elif 'fraude' in name:
                categories['🚨 Fraudes'].append(doc)
            else:
                categories['📁 Autres'].append(doc)
        
        response = f"📊 Analyse de {len(relevant_docs)} dossiers:\n\n"
        for cat, docs in sorted(categories.items()):
            response += f"**{cat}** ({len(docs)}):\n"
            for doc in docs[:5]:
                emoji = "🟢" if doc['score'] > 0.7 else "🟡" if doc['score'] > 0.5 else "🟠"
                response += f"  {emoji} {doc['name']} ({doc['score']:.2f})\n"
            response += "\n"
        
        return response.strip()

    def _vector_search_legacy(self, query: str, top_k=10, 
                             filters: Optional[Dict] = None) -> List[Dict]:
        """Legacy vector search"""
        if not self.use_rag or not self.embedding_model:
            return []
        
        try:
            query_vector = self.embedding_model.encode(query)
            return self._vector_search_direct(query_vector.tolist(), top_k, filters)
        except Exception as e:
            print(f"❌ Vector search error: {e}")
            return []

    def _process_basic(self, user_message: str, folders: List[Dict]) -> str:
        """Fallback"""
        msg_lower = user_message.lower()
        
        if any(word in msg_lower for word in ["bonjour", "salut", "hello"]):
            return f"👋 Bonjour ! LegalMind ULTRA-PRO + Multi-Modal.\n📁 {len(folders)} dossiers."
        
        if "combien" in msg_lower:
            return f"📊 {len(folders)} dossiers."
        
        return f"💼 LegalMind prêt ! {len(folders)} dossiers."

    def _format_sources_enhanced(self, docs: List[Dict]) -> str:
        """Format sources"""
        if not docs:
            return ""
        
        sources = "📚 **Sources:**\n"
        for i, doc in enumerate(docs, 1):
            score = doc['score']
            emoji = "🟢" if score > 0.8 else "🟡" if score > 0.6 else "🟠"
            quality = "Très pertinent" if score > 0.8 else "Pertinent" if score > 0.6 else "Moyen"
            
            search_type = doc.get('search_type', 'hybrid')
            sources += f"  [{i}] {emoji} {doc['name']}\n"
            sources += f"       └─ {quality} • {score:.3f} • {search_type}\n"
        
        return sources

    def get_metrics(self) -> Dict:
        """Métriques"""
        avg_time = np.mean(self.metrics['avg_response_time']) if self.metrics['avg_response_time'] else 0
        cache_rate = (self.metrics['cache_hits'] / max(1, self.metrics['total_queries'])) * 100
        
        return {
            'total_queries': self.metrics['total_queries'],
            'cache_hit_rate': f"{cache_rate:.1f}%",
            'avg_response_time': f"{avg_time:.2f}s",
            'weights': {
                'vector': f"{self.weights['vector']:.2f}",
                'fulltext': f"{self.weights['fulltext']:.2f}",
                'rerank_boost': f"{self.weights['rerank_boost']:.2f}"
            },
            'top_queries': dict(self.metrics['top_queries'].most_common(5))
        }

    def _generate_cache_key(self, query: str, filters: Optional[Dict] = None) -> str:
        key_data = f"{query}:{json.dumps(filters or {}, sort_keys=True)}"
        return f"chat:{hashlib.md5(key_data.encode()).hexdigest()}"

    async def _get_from_cache(self, key: str) -> Optional[str]:
        if not self.cache_enabled:
            return None
        try:
            value = await self.redis.get(key)
            return value.decode('utf-8') if value else None
        except:
            return None

    async def _store_in_cache(self, key: str, value: str, ttl: int = 3600):
        if not self.cache_enabled:
            return
        try:
            await self.redis.setex(key, ttl, value.encode('utf-8'))
        except Exception as e:
            print(f"⚠️ Cache error: {e}")

    async def _get_embedding_from_cache(self, key: str) -> Optional[List[float]]:
        if not self.cache_enabled:
            return None
        try:
            value = await self.redis.get(key)
            if value:
                return json.loads(value.decode('utf-8'))
        except:
            return None
        return None

    async def _store_embedding_in_cache(self, key: str, embedding: List[float], ttl: int = 86400):
        if not self.cache_enabled:
            return
        try:
            await self.redis.setex(key, ttl, json.dumps(embedding).encode('utf-8'))
        except Exception as e:
            print(f"⚠️ Embedding cache error: {e}")
