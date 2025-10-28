# Version améliorée sans API
class ChatHandler:
    def __init__(self, engine=None):
        self.engine = engine
        print("⚡ Mode basique (sans Claude API)")
    
    async def process_message(self, user_message: str, folders: List[Dict]) -> str:
        msg_lower = user_message.lower()
        
        # Recherche intelligente
        if "cherche" in msg_lower or "trouve" in msg_lower or "recherche" in msg_lower:
            keywords = [w for w in msg_lower.split() if len(w) > 3 and w not in ["cherche", "trouve", "recherche", "dans", "pour", "avec"]]
            results = []
            for folder in folders:
                name = folder.get("name", "").lower()
                if any(kw in name for kw in keywords):
                    results.append(folder["name"])
            
            if results:
                limited = results[:20]
                return f"✅ Trouvé {len(results)} dossiers:\n" + "\n".join([f"- {r}" for r in limited])
            return "❌ Aucun dossier trouvé pour ces critères."
        
        # Analyse par catégorie
        if "résume" in msg_lower or "aperçu" in msg_lower or "vue d'ensemble" in msg_lower:
            categories = {}
            for folder in folders:
                name = folder.get("name", "").lower()
                if "plainte" in name:
                    categories["Plaintes"] = categories.get("Plaintes", 0) + 1
                elif "argumentaire" in name:
                    categories["Argumentaires"] = categories.get("Argumentaires", 0) + 1
                elif "élément" in name:
                    categories["Éléments de preuve"] = categories.get("Éléments de preuve", 0) + 1
                elif "photo" in name:
                    categories["Photos/Documents"] = categories.get("Photos/Documents", 0) + 1
            
            summary = f"📊 Vue d'ensemble de vos {len(folders)} dossiers:\n\n"
            for cat, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
                summary += f"• {cat}: {count}\n"
            return summary
        
        # Statistiques
        if "combien" in msg_lower or "nombre" in msg_lower:
            query_word = None
            for word in ["plainte", "argumentaire", "photo", "élément"]:
                if word in msg_lower:
                    query_word = word
                    break
            
            if query_word:
                count = sum(1 for f in folders if query_word in f.get("name", "").lower())
                return f"📊 Vous avez {count} dossiers contenant '{query_word}'"
            return f"📊 Vous avez {len(folders)} dossiers au total."
        
        # Liste
        if "liste" in msg_lower:
            if "plainte" in msg_lower:
                plaintes = [f["name"] for f in folders if "plainte" in f.get("name", "").lower()][:20]
                return f"📋 Vos plaintes ({len(plaintes)} affichées):\n" + "\n".join([f"- {p}" for p in plaintes])
            else:
                return f"📋 {len(folders)} dossiers. Précisez ce que vous cherchez !"
        
        # Défaut
        return f"💡 J'ai accès à vos {len(folders)} dossiers. Essayez:\n- 'Cherche plainte'\n- 'Résume mes dossiers'\n- 'Combien de plaintes ?'"
