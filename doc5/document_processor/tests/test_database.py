"""
Tests for database operations
"""
import unittest
import json
import sqlite3
from unittest.mock import patch, MagicMock, mock_open

from flask import g

from document_processor.db.database import (
    get_db, close_db, init_db, save_processing_results,
    get_processing_results, save_correction, get_investments,
    find_investment_by_name
)
from tests.test_base import BaseTestCase

class TestDatabaseOperations(BaseTestCase):
    """Test cases for database operations"""
    
    def setUp(self):
        """Set up test environment"""
        super().setUp()
        
        # Create in-memory database for testing
        self.conn = sqlite3.connect(':memory:')
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        
        # Create tables
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS investments 
                              (id INTEGER PRIMARY KEY, name TEXT)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS processing_results 
                              (session_id TEXT PRIMARY KEY, file_path TEXT, 
                               results_json TEXT, corrections_json TEXT)''')
        
        # Insert test data
        self.cursor.execute("INSERT INTO investments (id, name) VALUES (1, 'Investment A')")
        self.cursor.execute("INSERT INTO investments (id, name) VALUES (2, 'Investment B')")
        self.cursor.execute("INSERT INTO investments (id, name) VALUES (3, 'ABC Partners')")
        
        test_results = {"summary": {"document_type": "K1"}, "text": "Sample text"}
        self.cursor.execute(
            "INSERT INTO processing_results (session_id, file_path, results_json) VALUES (?, ?, ?)",
            ("test-session", "test-file.pdf", json.dumps(test_results))
        )
        
        self.conn.commit()
        
        # Patch g object
        self.g_patch = patch.object(g, 'db', self.conn)
        self.g_patch.start()
    
    def tearDown(self):
        """Clean up test environment"""
        self.g_patch.stop()
        self.conn.close()
        super().tearDown()
    
    def test_get_db(self):
        """Test get_db function"""
        # Remove existing db from g
        if hasattr(g, 'db'):
            delattr(g, 'db')
        
        # Patch sqlite3.connect
        with patch('document_processor.db.database.sqlite3.connect') as mock_connect:
            mock_db = MagicMock()
            mock_connect.return_value = mock_db
            
            # Call get_db
            db = get_db()
            
            # Verify connection was created
            mock_connect.assert_called_once_with('investments.db', check_same_thread=False)
            
            # Verify row_factory was set
            self.assertEqual(mock_db.row_factory, sqlite3.Row)
            
            # Verify db was stored in g
            self.assertEqual(g.db, mock_db)
            
            # Call get_db again
            db2 = get_db()
            
            # Verify connection was reused
            self.assertEqual(mock_connect.call_count, 1)
            self.assertEqual(db, db2)
    
    def test_close_db(self):
        """Test close_db function"""
        # Create mock db
        mock_db = MagicMock()
        g.db = mock_db
        
        # Call close_db
        close_db()
        
        # Verify db was closed and removed from g
        mock_db.close.assert_called_once()
        self.assertFalse(hasattr(g, 'db'))
        
        # Test with no db in g
        close_db()  # Should not raise exception
    
    def test_init_db(self):
        """Test init_db function"""
        # Create mock app
        mock_app = MagicMock()
        mock_context = MagicMock()
        mock_app.app_context.return_value = mock_context
        
        # Call init_db
        with patch('document_processor.db.database.get_db') as mock_get_db:
            mock_db = MagicMock()
            mock_cursor = MagicMock()
            mock_db.cursor.return_value = mock_cursor
            mock_get_db.return_value = mock_db
            
            init_db(mock_app)
            
            # Verify app context was entered
            mock_app.app_context.assert_called_once()
            mock_context.__enter__.assert_called_once()
            
            # Verify tables were created
            self.assertEqual(mock_cursor.execute.call_count, 4)
            
            # Verify commit was called
            mock_db.commit.assert_called_once()
            
            # Verify teardown_appcontext was registered
            mock_app.teardown_appcontext.assert_called_once_with(close_db)
    
    def test_save_processing_results_new(self):
        """Test saving new processing results"""
        # Save new result with file path
        session_id = save_processing_results("new-session", file_path="new-file.pdf")
        
        # Verify result is in database
        self.cursor.execute(
            "SELECT file_path FROM processing_results WHERE session_id = ?",
            ("new-session",)
        )
        result = self.cursor.fetchone()
        
        self.assertIsNotNone(result)
        self.assertEqual(result['file_path'], "new-file.pdf")
        
        # Verify session ID was returned
        self.assertEqual(session_id, "new-session")
    
    def test_save_processing_results_update(self):
        """Test updating existing processing results"""
        # Update existing result with results
        test_results = {"summary": {"document_type": "Updated K1"}, "text": "Updated text"}
        session_id = save_processing_results("test-session", results=test_results)
        
        # Verify result was updated
        self.cursor.execute(
            "SELECT results_json FROM processing_results WHERE session_id = ?",
            ("test-session",)
        )
        result = self.cursor.fetchone()
        
        self.assertIsNotNone(result)
        updated_results = json.loads(result['results_json'])
        self.assertEqual(updated_results['summary']['document_type'], "Updated K1")
        
        # Verify session ID was returned
        self.assertEqual(session_id, "test-session")
    
    def test_get_processing_results(self):
        """Test retrieving processing results"""
        # Get existing results
        results = get_processing_results("test-session")
        
        # Verify results
        self.assertIsNotNone(results)
        self.assertEqual(results['summary']['document_type'], "K1")
        self.assertEqual(results['text'], "Sample text")
        
        # Test with non-existent session
        non_existent = get_processing_results("nonexistent-session")
        self.assertIsNone(non_existent)
    
    def test_save_correction(self):
        """Test saving corrections"""
        # Save correction
        corrections = {"correct_type": "Tax Return"}
        result = save_correction("test-session", corrections)
        
        # Verify result
        self.assertTrue(result)
        
        # Verify correction was saved
        self.cursor.execute(
            "SELECT corrections_json FROM processing_results WHERE session_id = ?",
            ("test-session",)
        )
        result = self.cursor.fetchone()
        
        self.assertIsNotNone(result)
        saved_corrections = json.loads(result['corrections_json'])
        self.assertEqual(saved_corrections['correct_type'], "Tax Return")
    
    def test_get_investments(self):
        """Test retrieving investments"""
        # Get investments
        investments = get_investments()
        
        # Verify investments
        self.assertEqual(len(investments), 3)
        self.assertEqual(investments[0]['id'], 1)
        self.assertEqual(investments[0]['name'], "Investment A")
        self.assertEqual(investments[1]['id'], 2)
        self.assertEqual(investments[1]['name'], "Investment B")
        self.assertEqual(investments[2]['id'], 3)
        self.assertEqual(investments[2]['name'], "ABC Partners")
    
    def test_find_investment_by_name(self):
        """Test finding investment by name"""
        # Find existing investment (exact match)
        investment = find_investment_by_name("Investment A")
        
        # Verify investment
        self.assertIsNotNone(investment)
        self.assertEqual(investment['id'], 1)
        self.assertEqual(investment['name'], "Investment A")
        
        # Find with partial match
        partial = find_investment_by_name("ABC")
        
        # Verify match
        self.assertIsNotNone(partial)
        self.assertEqual(partial['id'], 3)
        self.assertEqual(partial['name'], "ABC Partners")
        
        # Test with non-existent name
        non_existent = find_investment_by_name("Nonexistent")
        self.assertIsNone(non_existent)
        
        # Test with None
        none_result = find_investment_by_name(None)
        self.assertIsNone(none_result)

if __name__ == '__main__':
    unittest.main()