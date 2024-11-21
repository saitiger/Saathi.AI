import sqlite3
from datetime import datetime, timedelta
import requests
import schedule
import time
import logging
import os 

os.dotenv()

class MedicationReminderSystem:
    def __init__(self, vapi_api_key):
        """
        Initialize the medication reminder system
        
        :param vapi_api_key: API key for VAPI voice services
        """
        self.vapi_api_key = vapi_api_key
        self.setup_database()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s: %(message)s',
            filename='medication_reminders.log'
        )
    
    def setup_database(self):
        """
        Create SQLite database and tables for patient and medication tracking
        """
        try:
            self.conn = sqlite3.connect('medication_reminders.db')
            self.cursor = self.conn.cursor()
            
            # Create patients table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    phone_number TEXT NOT NULL,
                    language TEXT DEFAULT 'en'
                )
            ''')
            
            # Create medications table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS medications (
                    id INTEGER PRIMARY KEY,
                    patient_id INTEGER,
                    medication_name TEXT NOT NULL,
                    dosage TEXT NOT NULL,
                    frequency TEXT NOT NULL,
                    time_of_day TEXT NOT NULL,
                    FOREIGN KEY (patient_id) REFERENCES patients(id)
                )
            ''')
            
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Database setup error: {e}")
    
    def add_patient(self, name, phone_number, language='en'):
        """
        Add a new patient to the system
        
        :param name: Patient's full name
        :param phone_number: Patient's phone number
        :param language: Patient's preferred language
        :return: Patient ID
        """
        try:
            self.cursor.execute(
                'INSERT INTO patients (name, phone_number, language) VALUES (?, ?, ?)',
                (name, phone_number, language)
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            logging.error(f"Error adding patient: {e}")
            return None
    
    def add_medication(self, patient_id, medication_name, dosage, frequency, time_of_day):
        """
        Add a medication for a patient
        
        :param patient_id: ID of the patient
        :param medication_name: Name of the medication
        :param dosage: Medication dosage
        :param frequency: How often the medication should be taken
        :param time_of_day: Time medication should be taken
        :return: Medication ID
        """
        try:
            self.cursor.execute(
                '''INSERT INTO medications 
                (patient_id, medication_name, dosage, frequency, time_of_day) 
                VALUES (?, ?, ?, ?, ?)''',
                (patient_id, medication_name, dosage, frequency, time_of_day)
            )
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            logging.error(f"Error adding medication: {e}")
            return None
    
    def send_voice_reminder(self, patient_phone, patient_name, medication_details):
        """
        Send a voice reminder via VAPI
        
        :param patient_phone: Patient's phone number
        :param patient_name: Patient's name
        :param medication_details: Details of medication to take
        """
        try:
            # Example VAPI call (replace with actual VAPI endpoint)
            reminder_message = (
                f"Hello {patient_name}, this is a medication reminder. "
                f"Please take {medication_details['medication_name']} "
                f"with a dosage of {medication_details['dosage']}."
            )
            
            response = requests.post(
                'https://api.vapi.ai/calls',
                headers={
                    'Authorization': f'Bearer {self.vapi_api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'phoneNumber': patient_phone,
                    'message': reminder_message
                }
            )
            
            if response.status_code == 200:
                logging.info(f"Reminder sent to {patient_name}")
            else:
                logging.error(f"Failed to send reminder to {patient_name}")
        
        except requests.RequestException as e:
            logging.error(f"VAPI call failed: {e}")
    
    def check_medication_reminders(self):
        """
        Check and send medication reminders for current time
        """
        current_time = datetime.now().strftime('%H:%M')
        
        try:
            # Query patients with medications to take at current time
            self.cursor.execute('''
                SELECT p.name, p.phone_number, m.medication_name, m.dosage
                FROM patients p
                JOIN medications m ON p.id = m.patient_id
                WHERE m.time_of_day = ?
            ''', (current_time,))
            
            reminders = self.cursor.fetchall()
            
            for reminder in reminders:
                patient_name, phone_number, medication_name, dosage = reminder
                medication_details = {
                    'medication_name': medication_name,
                    'dosage': dosage
                }
                
                self.send_voice_reminder(phone_number, patient_name, medication_details)
        
        except sqlite3.Error as e:
            logging.error(f"Error checking reminders: {e}")
    
    def start_reminder_scheduler(self):
        """
        Start the medication reminder scheduler
        """
        schedule.every().minute.do(self.check_medication_reminders)
        
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    def __del__(self):
        """
        Close database connection when object is deleted
        """
        if hasattr(self, 'conn'):
            self.conn.close()

# Example usage
def main():
    reminder_system = MedicationReminderSystem(VAPI_API_KEY)
    
    # Add a sample patient
    patient_id = reminder_system.add_patient(
        name='John Doe', 
        phone_number='+15551234567'
    )
    
    # Add medication for the patient
    reminder_system.add_medication(
        patient_id=patient_id,
        medication_name='Metformin',
        dosage='500mg',
        frequency='daily',
        time_of_day='08:00'
    )
    
    # Start the reminder scheduler
    reminder_system.start_reminder_scheduler()

if __name__ == '__main__':
    main()
