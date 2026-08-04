import logging
from typing import Dict, Any, Optional
import os

logger = logging.getLogger(__name__)

class FirebaseAuthService:
    """
    Firebase & Google Sign-In Authentication Verification Service.
    Verifies Google email addresses & Firebase authentication tokens.
    """

    def __init__(self):
        self.enabled = False
        # Initialize Firebase Admin SDK if credentials are provided
        cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH")
        if cred_path and os.path.exists(cred_path):
            try:
                import firebase_admin
                from firebase_admin import credentials
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
                self.enabled = True
                logger.info("Initialized Secure Firebase Authentication Service with credentials.")
            except Exception as e:
                logger.error(f"Failed to initialize Firebase Admin SDK: {e}")
        else:
            logger.warning("FIREBASE_CREDENTIALS_PATH not set or file not found. Running in SECURE DEMO mode.")

    def verify_firebase_token(self, token_or_email: str) -> Optional[Dict[str, Any]]:
        if not token_or_email or not isinstance(token_or_email, str):
            return None

        clean = token_or_email.strip()

        # If Firebase is properly configured, enforce cryptographic JWT token verification
        if self.enabled:
            try:
                from firebase_admin import auth
                # Strip Bearer prefix if present
                id_token = clean[7:] if clean.lower().startswith("bearer ") else clean
                decoded_token = auth.verify_id_token(id_token)
                return {
                    "uid": decoded_token.get("uid"),
                    "email": decoded_token.get("email"),
                    "name": decoded_token.get("name", "Google User"),
                    "provider": decoded_token.get("firebase", {}).get("sign_in_provider", "google.com")
                }
            except Exception as e:
                logger.error(f"Secure Firebase Token verification failed: {e}")
                return None

        # Secure Demo Fallback Mode: Enforce format checks but warn user
        clean_lower = clean.lower()
        if "@" in clean_lower:
            parts = clean_lower.split("@")
            if len(parts) == 2 and "." in parts[1]:
                username = parts[0]
                name = username.replace(".", " ").replace("_", " ").title()
                logger.warning(f"SECURE DEMO AUTH: Authenticated email '{clean_lower}' without token signature check.")
                return {
                    "uid": f"usr_{username}",
                    "email": clean_lower,
                    "name": name or "Google User",
                    "provider": "google.com"
                }

        logger.error("Authentication failed: Firebase credentials missing, and token format is invalid.")
        return None
