"""
Transactions, supporting atomic multi-statement operations.
"""
import shoji


class Transaction:
    def __init__(self, wsm: shoji.WorkspaceManager) -> None:
        self.db = wsm._db
          
    def __enter__(self):
        self.db.transaction = self.db.create_transaction()
        return self
      
    def __exit__(self, exc_type, exc_value, exc_traceback): 
        if exc_type is None:
            self.db.transaction.commit().wait()
            self.db.transaction = self.db
            return True  # There was no exception
        else:
            self.db.transaction.reset()
            self.db.transaction = self.db
            return False  # Re-raise the exception
