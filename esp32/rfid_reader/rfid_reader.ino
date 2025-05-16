#include <SPI.h>
#include <MFRC522.h>

#define SS_PIN  5  // SDA Pin on RC522
#define RST_PIN 4  // RST Pin on RC522

MFRC522 mfrc522(SS_PIN, RST_PIN); // Create MFRC522 instance

void setup() {
  Serial.begin(115200); // Initialize serial communication
  while (!Serial);      // Wait for serial port to connect (needed for native USB)
  SPI.begin();          // Init SPI bus
  mfrc522.PCD_Init();   // Init MFRC522
  Serial.println("RFID Reader Initialized. Waiting for card...");
}

void loop() {
  // Look for new cards
  if ( ! mfrc522.PICC_IsNewCardPresent()) {
    return;
  }

  // Select one of the cards
  if ( ! mfrc522.PICC_ReadCardSerial()) {
    return;
  }

  // Dump UID
  Serial.print("RFID_UID:"); // Prefix to identify the data type
  String content = "";
  for (byte i = 0; i < mfrc522.uid.size; i++) {
     content.concat(String(mfrc522.uid.uidByte[i] < 0x10 ? " 0" : " "));
     content.concat(String(mfrc522.uid.uidByte[i], HEX));
  }
  content.toUpperCase();
  content.trim(); // Remove leading/trailing whitespace
  Serial.println(content); // Send UID over serial

  mfrc522.PICC_HaltA(); // Halt PICC
  mfrc522.PCD_StopCrypto1(); // Stop encryption on PCD
  delay(1000); // Delay a bit before next read
}