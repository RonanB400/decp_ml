#!/usr/bin/env python3
"""
Example usage of the RAGQuerySystem module.

This script demonstrates how to use the improved RAG Query System
for natural language database queries.
"""

import os
import sys

# Add scripts directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import after path modification
from rag_query import RAGQuerySystem


def main():
    """Demonstrate usage of RAGQuerySystem."""
    print("🚀 RAG Query System Example")
    print("=" * 40)
    
    try:
        # Initialize the system with custom parameters
        print("📡 Initializing RAG Query System...")
        rag_system = RAGQuerySystem(
            db_path=None,  # Uses default path
            top_k=5        # Limit results to 5
        )
        
        # Test database connection
        print("🔌 Testing database connection...")
        if not rag_system.test_connection():
            print("❌ Failed to connect to database")
            return
        print("✅ Database connection successful!")
        
        # Get database information
        print("\n📊 Database Information:")
        db_info = rag_system.get_database_info()
        print(f"   Dialect: {db_info['dialect']}")
        print(f"   Tables: {db_info['tables']}")
        
        # Example queries
        questions = [
            "How many rows are there in the table?",
            "What are the first 3 entries in the dataset?",
            "Show me some sample data from the table"
        ]
        
        print("\n🤖 Running example queries...")
        print("-" * 40)
        
        for i, question in enumerate(questions, 1):
            print(f"\n📝 Query {i}: {question}")
            
            # Process the query
            result = rag_system.query(question)
            
            # Display results
            print(f"🔍 Generated SQL: {result.get('query', 'N/A')}")
            print(f"📊 Database Result: {result.get('result', 'N/A')}")
            print(f"💬 AI Answer: {result.get('answer', 'N/A')}")
            print("-" * 40)
        
        print("\n✨ Example completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Database file not found: {e}")
        print("💡 Make sure the database file exists at the expected location")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Check your API key and database configuration")


if __name__ == "__main__":
    main() 