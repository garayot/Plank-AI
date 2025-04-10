from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QScrollArea, QLineEdit, QPushButton, QLabel,
    QSizePolicy, QHBoxLayout, QTextEdit, QWidget
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont
import google.generativeai as genai
from dotenv import load_dotenv
import os

class AIChatDialog(QDialog):
    response_received = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Plank.AI Assistant")
        self.setFixedSize(350, 500)
        self.setStyleSheet("""
            QDialog {
                background-color: #1e1e1e;
                border-radius: 12px;
                padding: 10px;
            }
            QTextEdit, QLineEdit {
                background-color: #2e2e2e;
                color: white;
                border: 1px solid #444;
                border-radius: 6px;
                font-size: 13px;
                padding: 8px;
            }
            QPushButton {
                background-color: #3a82f7;
                color: white;
                border-radius: 6px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #5a9af9;
            }
        """)

        # AI Client setup
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")
        self.chat = self.model.start_chat(history=[])

        # Main layout
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)

        # Close button
        close_btn = QPushButton("×")
        close_btn.setFixedSize(24, 24)
        close_btn.clicked.connect(self.hide)
        close_btn.setStyleSheet("background-color: transparent; font-size: 18px; color: white; border: none;")
        close_btn.setCursor(Qt.PointingHandCursor)

        top_bar = QHBoxLayout()
        top_bar.addStretch()
        top_bar.addWidget(close_btn)
        self.layout.addLayout(top_bar)

        # Chat history
        self.chat_history = QScrollArea()
        self.chat_history.setWidgetResizable(True)
        self.history_content = QWidget()
        self.history_layout = QVBoxLayout(self.history_content)
        self.chat_history.setWidget(self.history_content)
        self.layout.addWidget(self.chat_history)

        # Input area
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask about the data...")
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.process_query)

        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)
        self.layout.addLayout(input_layout)

        self.response_received.connect(self.display_response)

    def process_query(self):
        query = self.input_field.text()
        if not query:
            return

        self.add_message(query, is_user=True)
        self.input_field.clear()

        # Add context if available
        tally_data = self.parent.species_tally if self.parent else {}
        context = f"Current plankton tally: {tally_data}. "
        QTimer.singleShot(100, lambda: self.get_ai_response(context + query))

    def get_ai_response(self, query):
        try:
            response = self.chat.send_message(query)
            self.response_received.emit(response.text)
        except Exception as e:
            self.response_received.emit(f"Error: {str(e)}")

    def display_response(self, response):
        self.add_message(response, is_user=False)

    def add_message(self, text, is_user=True):
        message = QLabel(text)
        message.setWordWrap(True)
        message.setStyleSheet(f"""
            background-color: {'#3498db' if is_user else '#e0e0e0'};
            color: {'white' if is_user else 'black'};
            border-radius: 8px;
            padding: 8px;

        """)
        message.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        self.history_layout.addWidget(message, 0, Qt.AlignRight if is_user else Qt.AlignLeft)

        QTimer.singleShot(50, lambda: self.chat_history.verticalScrollBar().setValue(
            self.chat_history.verticalScrollBar().maximum()
        ))
