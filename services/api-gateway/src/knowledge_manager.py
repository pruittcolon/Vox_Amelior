"""
Knowledge Manager Module - Enterprise Knowledge Management
Handles knowledge base articles, topics taxonomy, and expertise finder.
"""

import json
import logging
import sqlite3
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ArticleStatus(str, Enum):
    """Article publication status."""

    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class ProficiencyLevel(str, Enum):
    """Skill proficiency levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class KnowledgeManager:
    """
    Enterprise knowledge management engine.

    Features:
    - Knowledge base with articles and topics
    - Full-text search with ranking
    - Expertise finder with skill profiling
    - Universal search aggregation
    """

    def __init__(self, db_path: str = "/app/instance/knowledge_store.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()
        logger.info(f"KnowledgeManager initialized with db: {self.db_path}")

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript("""
            -- Topics table (hierarchical taxonomy)
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                parent_id INTEGER,
                icon TEXT,
                sort_order INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (parent_id) REFERENCES topics(id) ON DELETE SET NULL
            );
            
            -- Articles table
            CREATE TABLE IF NOT EXISTS articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                summary TEXT,
                author_id TEXT,
                author_name TEXT,
                status TEXT DEFAULT 'draft',
                tags TEXT,  -- JSON array
                view_count INTEGER DEFAULT 0,
                rating_sum REAL DEFAULT 0,
                rating_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT,
                published_at TEXT
            );
            
            -- Article-Topic mapping
            CREATE TABLE IF NOT EXISTS article_topics (
                article_id INTEGER NOT NULL,
                topic_id INTEGER NOT NULL,
                PRIMARY KEY (article_id, topic_id),
                FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE,
                FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
            );
            
            -- Article ratings
            CREATE TABLE IF NOT EXISTS article_ratings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_id INTEGER NOT NULL,
                user_id TEXT NOT NULL,
                rating INTEGER NOT NULL CHECK(rating >= 1 AND rating <= 5),
                created_at TEXT NOT NULL,
                FOREIGN KEY (article_id) REFERENCES articles(id) ON DELETE CASCADE,
                UNIQUE(article_id, user_id)
            );
            
            -- Skills catalog
            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                category TEXT,
                description TEXT,
                created_at TEXT NOT NULL
            );
            
            -- Expert profiles
            CREATE TABLE IF NOT EXISTS expert_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                bio TEXT,
                title TEXT,
                department TEXT,
                contact_preference TEXT,  -- email, slack, etc.
                is_available INTEGER DEFAULT 1,
                articles_count INTEGER DEFAULT 0,
                answers_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT
            );
            
            -- Expert skills (many-to-many with proficiency)
            CREATE TABLE IF NOT EXISTS expert_skills (
                expert_id INTEGER NOT NULL,
                skill_id INTEGER NOT NULL,
                proficiency TEXT DEFAULT 'intermediate',
                years_experience REAL,
                certified INTEGER DEFAULT 0,
                PRIMARY KEY (expert_id, skill_id),
                FOREIGN KEY (expert_id) REFERENCES expert_profiles(id) ON DELETE CASCADE,
                FOREIGN KEY (skill_id) REFERENCES skills(id) ON DELETE CASCADE
            );
            
            -- Expert-Topic mapping
            CREATE TABLE IF NOT EXISTS expert_topics (
                expert_id INTEGER NOT NULL,
                topic_id INTEGER NOT NULL,
                PRIMARY KEY (expert_id, topic_id),
                FOREIGN KEY (expert_id) REFERENCES expert_profiles(id) ON DELETE CASCADE,
                FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
            );
            
            -- Search query log for analytics
            CREATE TABLE IF NOT EXISTS search_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                source TEXT,  -- kb, experts, universal
                results_count INTEGER,
                user_id TEXT,
                created_at TEXT NOT NULL
            );
            
            -- Full-text search index for articles
            CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
                title, content, summary, tags,
                content='articles',
                content_rowid='id'
            );
            
            -- Triggers to keep FTS index in sync
            CREATE TRIGGER IF NOT EXISTS articles_ai AFTER INSERT ON articles BEGIN
                INSERT INTO articles_fts(rowid, title, content, summary, tags)
                VALUES (new.id, new.title, new.content, new.summary, new.tags);
            END;
            
            CREATE TRIGGER IF NOT EXISTS articles_ad AFTER DELETE ON articles BEGIN
                INSERT INTO articles_fts(articles_fts, rowid, title, content, summary, tags)
                VALUES ('delete', old.id, old.title, old.content, old.summary, old.tags);
            END;
            
            CREATE TRIGGER IF NOT EXISTS articles_au AFTER UPDATE ON articles BEGIN
                INSERT INTO articles_fts(articles_fts, rowid, title, content, summary, tags)
                VALUES ('delete', old.id, old.title, old.content, old.summary, old.tags);
                INSERT INTO articles_fts(rowid, title, content, summary, tags)
                VALUES (new.id, new.title, new.content, new.summary, new.tags);
            END;
            
            -- Indexes
            CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);
            CREATE INDEX IF NOT EXISTS idx_articles_author ON articles(author_id);
            CREATE INDEX IF NOT EXISTS idx_expert_profiles_user ON expert_profiles(user_id);
            CREATE INDEX IF NOT EXISTS idx_search_logs_query ON search_logs(query);
        """)
        conn.commit()

        # Seed default topics and skills if empty
        self._seed_defaults()

    def _seed_defaults(self):
        """Seed default topics and skills if tables are empty."""
        conn = self._get_conn()

        # Check if topics exist
        topic_count = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
        if topic_count == 0:
            now = datetime.utcnow().isoformat() + "Z"
            default_topics = [
                (1, "Technical", "Technical documentation and guides", None, "⚙️", 1, now),
                (2, "Development", "Software development", 1, "💻", 1, now),
                (3, "Infrastructure", "DevOps and infrastructure", 1, "🏗️", 2, now),
                (4, "Security", "Security best practices", 1, "🔒", 3, now),
                (5, "Process", "Business processes and workflows", None, "📋", 2, now),
                (6, "Onboarding", "New employee resources", 5, "👋", 1, now),
                (7, "Best Practices", "Standards and guidelines", 5, "✅", 2, now),
            ]
            conn.executemany(
                "INSERT OR IGNORE INTO topics (id, name, description, parent_id, icon, sort_order, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                default_topics,
            )
            conn.commit()

        # Check if skills exist
        skill_count = conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
        if skill_count == 0:
            now = datetime.utcnow().isoformat() + "Z"
            default_skills = [
                ("Python", "Development", "Python programming language"),
                ("JavaScript", "Development", "JavaScript/TypeScript"),
                ("FastAPI", "Development", "FastAPI web framework"),
                ("Docker", "Infrastructure", "Containerization with Docker"),
                ("Kubernetes", "Infrastructure", "Container orchestration"),
                ("PostgreSQL", "Database", "PostgreSQL database"),
                ("Machine Learning", "AI/ML", "ML and data science"),
                ("API Design", "Architecture", "RESTful API design"),
                ("Project Management", "Business", "Project and team management"),
            ]
            conn.executemany(
                f"INSERT OR IGNORE INTO skills (name, category, description, created_at) VALUES (?, ?, ?, '{now}')",
                default_skills,
            )
            conn.commit()

    # =========================================================================
    # Topics CRUD
    # =========================================================================

    def list_topics(self, include_children: bool = True) -> list[dict[str, Any]]:
        """List all topics with optional hierarchy."""
        conn = self._get_conn()
        rows = conn.execute("SELECT * FROM topics ORDER BY parent_id NULLS FIRST, sort_order, name").fetchall()

        topics = [self._parse_topic_row(r) for r in rows]

        if include_children:
            # Build hierarchy
            topic_map = {t["id"]: t for t in topics}
            root_topics = []
            for t in topics:
                t["children"] = []
                if t["parent_id"] and t["parent_id"] in topic_map:
                    topic_map[t["parent_id"]]["children"].append(t)
                else:
                    root_topics.append(t)
            return root_topics

        return topics

    def _parse_topic_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "parent_id": row["parent_id"],
            "icon": row["icon"],
            "sort_order": row["sort_order"],
        }

    def create_topic(
        self,
        name: str,
        description: str | None = None,
        parent_id: int | None = None,
        icon: str | None = None,
    ) -> dict[str, Any]:
        """Create a new topic."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"

        cursor = conn.execute(
            "INSERT INTO topics (name, description, parent_id, icon, created_at) VALUES (?, ?, ?, ?, ?)",
            (name, description, parent_id, icon, now),
        )
        conn.commit()

        return {"id": cursor.lastrowid, "name": name, "description": description, "parent_id": parent_id}

    def get_topic(self, topic_id: int) -> dict[str, Any] | None:
        """Get a topic by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
        if not row:
            return None
        return self._parse_topic_row(row)

    # =========================================================================
    # Articles CRUD
    # =========================================================================

    def create_article(
        self,
        title: str,
        content: str,
        summary: str | None = None,
        topic_ids: list[int] | None = None,
        tags: list[str] | None = None,
        author_id: str | None = None,
        author_name: str | None = None,
        status: str = "draft",
    ) -> dict[str, Any]:
        """Create a new knowledge base article."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"
        published_at = now if status == "published" else None

        cursor = conn.execute(
            """
            INSERT INTO articles (title, content, summary, author_id, author_name, status, tags, created_at, published_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (title, content, summary, author_id, author_name, status, json.dumps(tags or []), now, published_at),
        )
        conn.commit()
        article_id = cursor.lastrowid

        # Add topic mappings
        if topic_ids:
            for tid in topic_ids:
                conn.execute(
                    "INSERT OR IGNORE INTO article_topics (article_id, topic_id) VALUES (?, ?)", (article_id, tid)
                )
            conn.commit()

        logger.info(f"Article created: id={article_id}, title={title}")
        return self.get_article(article_id)

    def get_article(self, article_id: int) -> dict[str, Any] | None:
        """Get an article by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        if not row:
            return None

        article = self._parse_article_row(row)

        # Get topics
        topic_rows = conn.execute(
            "SELECT t.* FROM topics t JOIN article_topics at ON t.id = at.topic_id WHERE at.article_id = ?",
            (article_id,),
        ).fetchall()
        article["topics"] = [self._parse_topic_row(r) for r in topic_rows]

        return article

    def _parse_article_row(self, row: sqlite3.Row) -> dict[str, Any]:
        rating = row["rating_sum"] / max(row["rating_count"], 1)
        return {
            "id": row["id"],
            "title": row["title"],
            "content": row["content"],
            "summary": row["summary"],
            "author_id": row["author_id"],
            "author_name": row["author_name"],
            "status": row["status"],
            "tags": json.loads(row["tags"] or "[]"),
            "view_count": row["view_count"],
            "rating": round(rating, 1),
            "rating_count": row["rating_count"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "published_at": row["published_at"],
        }

    def list_articles(
        self,
        topic_id: int | None = None,
        tag: str | None = None,
        status: str = "published",
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List articles with optional filters."""
        conn = self._get_conn()

        query = "SELECT DISTINCT a.* FROM articles a"
        params = []
        where_clauses = []

        if topic_id:
            query += " JOIN article_topics at ON a.id = at.article_id"
            where_clauses.append("at.topic_id = ?")
            params.append(topic_id)

        if status:
            where_clauses.append("a.status = ?")
            params.append(status)

        if tag:
            where_clauses.append("a.tags LIKE ?")
            params.append(f'%"{tag}"%')

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY a.published_at DESC, a.created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = conn.execute(query, params).fetchall()
        return [self._parse_article_row(r) for r in rows]

    def update_article(
        self,
        article_id: int,
        title: str | None = None,
        content: str | None = None,
        summary: str | None = None,
        topic_ids: list[int] | None = None,
        tags: list[str] | None = None,
        status: str | None = None,
    ) -> dict[str, Any] | None:
        """Update an article."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"

        updates = ["updated_at = ?"]
        params = [now]

        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if content is not None:
            updates.append("content = ?")
            params.append(content)
        if summary is not None:
            updates.append("summary = ?")
            params.append(summary)
        if tags is not None:
            updates.append("tags = ?")
            params.append(json.dumps(tags))
        if status is not None:
            updates.append("status = ?")
            params.append(status)
            if status == "published":
                updates.append("published_at = COALESCE(published_at, ?)")
                params.append(now)

        params.append(article_id)
        conn.execute(f"UPDATE articles SET {', '.join(updates)} WHERE id = ?", params)
        conn.commit()

        # Update topic mappings
        if topic_ids is not None:
            conn.execute("DELETE FROM article_topics WHERE article_id = ?", (article_id,))
            for tid in topic_ids:
                conn.execute(
                    "INSERT OR IGNORE INTO article_topics (article_id, topic_id) VALUES (?, ?)", (article_id, tid)
                )
            conn.commit()

        return self.get_article(article_id)

    def delete_article(self, article_id: int) -> dict[str, Any]:
        """Delete an article."""
        conn = self._get_conn()
        result = conn.execute("DELETE FROM articles WHERE id = ?", (article_id,))
        conn.commit()

        if result.rowcount == 0:
            return {"success": False, "error": "Article not found"}
        return {"success": True, "message": f"Article {article_id} deleted"}

    def rate_article(self, article_id: int, user_id: str, rating: int) -> dict[str, Any]:
        """Rate an article (1-5 stars)."""
        if rating < 1 or rating > 5:
            return {"success": False, "error": "Rating must be between 1 and 5"}

        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"

        # Check for existing rating
        existing = conn.execute(
            "SELECT rating FROM article_ratings WHERE article_id = ? AND user_id = ?", (article_id, user_id)
        ).fetchone()

        if existing:
            old_rating = existing["rating"]
            conn.execute(
                "UPDATE article_ratings SET rating = ?, created_at = ? WHERE article_id = ? AND user_id = ?",
                (rating, now, article_id, user_id),
            )
            conn.execute(
                "UPDATE articles SET rating_sum = rating_sum - ? + ? WHERE id = ?", (old_rating, rating, article_id)
            )
        else:
            conn.execute(
                "INSERT INTO article_ratings (article_id, user_id, rating, created_at) VALUES (?, ?, ?, ?)",
                (article_id, user_id, rating, now),
            )
            conn.execute(
                "UPDATE articles SET rating_sum = rating_sum + ?, rating_count = rating_count + 1 WHERE id = ?",
                (rating, article_id),
            )

        conn.commit()

        # Get updated rating
        article = self.get_article(article_id)
        return {"success": True, "new_rating": article["rating"], "rating_count": article["rating_count"]}

    def increment_view_count(self, article_id: int):
        """Increment article view count."""
        conn = self._get_conn()
        conn.execute("UPDATE articles SET view_count = view_count + 1 WHERE id = ?", (article_id,))
        conn.commit()

    # =========================================================================
    # Knowledge Base Search
    # =========================================================================

    def search_articles(
        self,
        query: str,
        topic_id: int | None = None,
        limit: int = 20,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Full-text search on knowledge base articles."""
        conn = self._get_conn()

        # Log search
        now = datetime.utcnow().isoformat() + "Z"

        # Use FTS5 for search
        search_query = """
            SELECT a.*, bm25(articles_fts) as rank
            FROM articles a
            JOIN articles_fts ON a.id = articles_fts.rowid
            WHERE articles_fts MATCH ? AND a.status = 'published'
        """
        params = [query]

        if topic_id:
            search_query = """
                SELECT a.*, bm25(articles_fts) as rank
                FROM articles a
                JOIN articles_fts ON a.id = articles_fts.rowid
                JOIN article_topics at ON a.id = at.article_id
                WHERE articles_fts MATCH ? AND a.status = 'published' AND at.topic_id = ?
            """
            params.append(topic_id)

        search_query += " ORDER BY rank LIMIT ?"
        params.append(limit)

        try:
            rows = conn.execute(search_query, params).fetchall()
        except sqlite3.OperationalError:
            # Fallback to LIKE search if FTS fails
            like_query = f"%{query}%"
            rows = conn.execute(
                "SELECT * FROM articles WHERE status = 'published' AND (title LIKE ? OR content LIKE ?) LIMIT ?",
                (like_query, like_query, limit),
            ).fetchall()

        results = [self._parse_article_row(r) for r in rows]

        # Log search
        conn.execute(
            "INSERT INTO search_logs (query, source, results_count, user_id, created_at) VALUES (?, 'kb', ?, ?, ?)",
            (query, len(results), user_id, now),
        )
        conn.commit()

        return results

    # =========================================================================
    # Skills CRUD
    # =========================================================================

    def list_skills(self, category: str | None = None) -> list[dict[str, Any]]:
        """List all skills."""
        conn = self._get_conn()

        if category:
            rows = conn.execute("SELECT * FROM skills WHERE category = ? ORDER BY name", (category,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM skills ORDER BY category, name").fetchall()

        return [
            {"id": r["id"], "name": r["name"], "category": r["category"], "description": r["description"]} for r in rows
        ]

    def create_skill(
        self,
        name: str,
        category: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Create a new skill."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"

        cursor = conn.execute(
            "INSERT INTO skills (name, category, description, created_at) VALUES (?, ?, ?, ?)",
            (name, category, description, now),
        )
        conn.commit()

        return {"id": cursor.lastrowid, "name": name, "category": category}

    def get_skill_by_name(self, name: str) -> dict[str, Any] | None:
        """Get skill by name."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM skills WHERE LOWER(name) = LOWER(?)", (name,)).fetchone()
        if not row:
            return None
        return {"id": row["id"], "name": row["name"], "category": row["category"]}

    # =========================================================================
    # Expert Profiles
    # =========================================================================

    def get_or_create_expert(self, user_id: str, name: str) -> dict[str, Any]:
        """Get or create an expert profile."""
        conn = self._get_conn()

        row = conn.execute("SELECT * FROM expert_profiles WHERE user_id = ?", (user_id,)).fetchone()
        if row:
            return self._parse_expert_row(row)

        now = datetime.utcnow().isoformat() + "Z"
        cursor = conn.execute(
            "INSERT INTO expert_profiles (user_id, name, created_at) VALUES (?, ?, ?)", (user_id, name, now)
        )
        conn.commit()

        return self.get_expert(cursor.lastrowid)

    def get_expert(self, expert_id: int) -> dict[str, Any] | None:
        """Get expert profile by ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM expert_profiles WHERE id = ?", (expert_id,)).fetchone()
        if not row:
            return None
        return self._parse_expert_row(row)

    def get_expert_by_user_id(self, user_id: str) -> dict[str, Any] | None:
        """Get expert profile by user ID."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM expert_profiles WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            return None
        return self._parse_expert_row(row)

    def _parse_expert_row(self, row: sqlite3.Row) -> dict[str, Any]:
        conn = self._get_conn()
        expert_id = row["id"]

        # Get skills
        skill_rows = conn.execute(
            """
            SELECT s.id, s.name, s.category, es.proficiency, es.years_experience, es.certified
            FROM skills s
            JOIN expert_skills es ON s.id = es.skill_id
            WHERE es.expert_id = ?
            ORDER BY es.proficiency DESC, s.name
            """,
            (expert_id,),
        ).fetchall()

        skills = [
            {
                "id": r["id"],
                "name": r["name"],
                "category": r["category"],
                "proficiency": r["proficiency"],
                "years_experience": r["years_experience"],
                "certified": bool(r["certified"]),
            }
            for r in skill_rows
        ]

        # Get topics
        topic_rows = conn.execute(
            "SELECT t.* FROM topics t JOIN expert_topics et ON t.id = et.topic_id WHERE et.expert_id = ?", (expert_id,)
        ).fetchall()
        topics = [self._parse_topic_row(r) for r in topic_rows]

        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "name": row["name"],
            "bio": row["bio"],
            "title": row["title"],
            "department": row["department"],
            "contact_preference": row["contact_preference"],
            "is_available": bool(row["is_available"]),
            "articles_count": row["articles_count"],
            "answers_count": row["answers_count"],
            "skills": skills,
            "topics": topics,
            "created_at": row["created_at"],
        }

    def update_expert_profile(
        self,
        user_id: str,
        name: str | None = None,
        bio: str | None = None,
        title: str | None = None,
        department: str | None = None,
        contact_preference: str | None = None,
        is_available: bool | None = None,
        skill_updates: list[dict[str, Any]] | None = None,
        topic_ids: list[int] | None = None,
    ) -> dict[str, Any] | None:
        """Update expert profile."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"

        # Get or create profile
        profile = self.get_expert_by_user_id(user_id)
        if not profile:
            profile = self.get_or_create_expert(user_id, name or "Unknown")

        expert_id = profile["id"]

        updates = ["updated_at = ?"]
        params = [now]

        if name is not None:
            updates.append("name = ?")
            params.append(name)
        if bio is not None:
            updates.append("bio = ?")
            params.append(bio)
        if title is not None:
            updates.append("title = ?")
            params.append(title)
        if department is not None:
            updates.append("department = ?")
            params.append(department)
        if contact_preference is not None:
            updates.append("contact_preference = ?")
            params.append(contact_preference)
        if is_available is not None:
            updates.append("is_available = ?")
            params.append(1 if is_available else 0)

        params.append(expert_id)
        conn.execute(f"UPDATE expert_profiles SET {', '.join(updates)} WHERE id = ?", params)

        # Update skills
        if skill_updates is not None:
            conn.execute("DELETE FROM expert_skills WHERE expert_id = ?", (expert_id,))
            for skill_data in skill_updates:
                skill = self.get_skill_by_name(skill_data.get("name", ""))
                if not skill:
                    skill = self.create_skill(skill_data["name"], skill_data.get("category"))
                conn.execute(
                    "INSERT INTO expert_skills (expert_id, skill_id, proficiency, years_experience, certified) VALUES (?, ?, ?, ?, ?)",
                    (
                        expert_id,
                        skill["id"],
                        skill_data.get("proficiency", "intermediate"),
                        skill_data.get("years_experience"),
                        1 if skill_data.get("certified") else 0,
                    ),
                )

        # Update topics
        if topic_ids is not None:
            conn.execute("DELETE FROM expert_topics WHERE expert_id = ?", (expert_id,))
            for tid in topic_ids:
                conn.execute("INSERT INTO expert_topics (expert_id, topic_id) VALUES (?, ?)", (expert_id, tid))

        conn.commit()
        return self.get_expert(expert_id)

    def list_experts(
        self,
        skill_name: str | None = None,
        topic_id: int | None = None,
        available_only: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List experts with filters."""
        conn = self._get_conn()

        query = "SELECT DISTINCT ep.* FROM expert_profiles ep"
        params = []
        where_clauses = []

        if skill_name:
            query += " JOIN expert_skills es ON ep.id = es.expert_id JOIN skills s ON es.skill_id = s.id"
            where_clauses.append("LOWER(s.name) = LOWER(?)")
            params.append(skill_name)

        if topic_id:
            query += " JOIN expert_topics et ON ep.id = et.expert_id"
            where_clauses.append("et.topic_id = ?")
            params.append(topic_id)

        if available_only:
            where_clauses.append("ep.is_available = 1")

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        query += " ORDER BY ep.articles_count + ep.answers_count DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [self._parse_expert_row(r) for r in rows]

    def find_experts(self, query: str, limit: int = 10, user_id: str | None = None) -> list[dict[str, Any]]:
        """Find experts by query (searches name, bio, skills, topics)."""
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"

        like_query = f"%{query}%"

        rows = conn.execute(
            """
            SELECT DISTINCT ep.* FROM expert_profiles ep
            LEFT JOIN expert_skills es ON ep.id = es.expert_id
            LEFT JOIN skills s ON es.skill_id = s.id
            LEFT JOIN expert_topics et ON ep.id = et.expert_id
            LEFT JOIN topics t ON et.topic_id = t.id
            WHERE ep.name LIKE ? OR ep.bio LIKE ? OR ep.title LIKE ?
                  OR s.name LIKE ? OR t.name LIKE ?
            ORDER BY ep.articles_count + ep.answers_count DESC
            LIMIT ?
            """,
            (like_query, like_query, like_query, like_query, like_query, limit),
        ).fetchall()

        results = [self._parse_expert_row(r) for r in rows]

        # Log search
        conn.execute(
            "INSERT INTO search_logs (query, source, results_count, user_id, created_at) VALUES (?, 'experts', ?, ?, ?)",
            (query, len(results), user_id, now),
        )
        conn.commit()

        return results

    # =========================================================================
    # Universal Search
    # =========================================================================

    def universal_search(
        self,
        query: str,
        sources: list[str] | None = None,
        limit: int = 20,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Search across multiple sources.

        Sources: 'kb' (knowledge base), 'experts'
        """
        if sources is None:
            sources = ["kb", "experts"]

        results = {}
        total = 0

        if "kb" in sources:
            kb_results = self.search_articles(query, limit=limit, user_id=user_id)
            results["articles"] = kb_results
            total += len(kb_results)

        if "experts" in sources:
            expert_results = self.find_experts(query, limit=limit, user_id=user_id)
            results["experts"] = expert_results
            total += len(expert_results)

        # Log universal search
        conn = self._get_conn()
        now = datetime.utcnow().isoformat() + "Z"
        conn.execute(
            "INSERT INTO search_logs (query, source, results_count, user_id, created_at) VALUES (?, 'universal', ?, ?, ?)",
            (query, total, user_id, now),
        )
        conn.commit()

        return {
            "query": query,
            "total_results": total,
            "sources": sources,
            "results": results,
        }

    def get_search_suggestions(self, partial: str, limit: int = 10) -> list[str]:
        """Get search suggestions based on previous queries."""
        conn = self._get_conn()

        like_query = f"{partial}%"
        rows = conn.execute(
            """
            SELECT query, COUNT(*) as freq FROM search_logs
            WHERE query LIKE ?
            GROUP BY query
            ORDER BY freq DESC
            LIMIT ?
            """,
            (like_query, limit),
        ).fetchall()

        return [r["query"] for r in rows]

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Get knowledge management statistics."""
        conn = self._get_conn()

        articles_total = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        articles_published = conn.execute("SELECT COUNT(*) FROM articles WHERE status = 'published'").fetchone()[0]
        topics_count = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
        experts_count = conn.execute("SELECT COUNT(*) FROM expert_profiles").fetchone()[0]
        skills_count = conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]

        # Recent search stats
        searches_24h = conn.execute(
            "SELECT COUNT(*) FROM search_logs WHERE created_at > datetime('now', '-1 day')"
        ).fetchone()[0]

        top_searches = conn.execute(
            """
            SELECT query, COUNT(*) as freq FROM search_logs
            WHERE created_at > datetime('now', '-7 day')
            GROUP BY query ORDER BY freq DESC LIMIT 5
            """
        ).fetchall()

        return {
            "articles": {
                "total": articles_total,
                "published": articles_published,
            },
            "topics": topics_count,
            "experts": experts_count,
            "skills": skills_count,
            "searches_24h": searches_24h,
            "top_searches": [{"query": r["query"], "count": r["freq"]} for r in top_searches],
        }


# Singleton instance
_knowledge_manager: KnowledgeManager | None = None


def get_knowledge_manager(db_path: str | None = None) -> KnowledgeManager:
    """Get or create the singleton KnowledgeManager instance."""
    global _knowledge_manager
    if _knowledge_manager is None:
        _knowledge_manager = KnowledgeManager(db_path or "/app/instance/knowledge_store.db")
    return _knowledge_manager
