import os
import unittest
import time
from fastapi import HTTPException
import jwt

# Ensure environment is set to development for base tests
os.environ["ENVIRONMENT"] = "development"
os.environ["DEBUG"] = "true"

from app.auth import get_current_user, is_dev_environment, AuthenticatedUser

class TestAuthMigration(unittest.TestCase):

    def setUp(self):
        os.environ["ENVIRONMENT"] = "development"
        os.environ["DEBUG"] = "true"

    def test_missing_auth_header_raises_401_with_devlens_branding(self):
        with self.assertRaises(HTTPException) as ctx:
            get_current_user(None)
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Please log in to DevLens", ctx.exception.detail)
        self.assertNotIn("HACKORA", ctx.exception.detail)

    def test_invalid_bearer_format_raises_401(self):
        with self.assertRaises(HTTPException) as ctx:
            get_current_user("Basic 123456")
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Please log in to DevLens", ctx.exception.detail)

    def test_empty_token_raises_401(self):
        with self.assertRaises(HTTPException) as ctx:
            get_current_user("Bearer    ")
        self.assertEqual(ctx.exception.status_code, 401)
        self.assertIn("Authorization token is empty", ctx.exception.detail)

    def test_devlens_dev_token_accepted_in_dev_environment(self):
        user = get_current_user("Bearer devlens_dev_test123456789")
        self.assertIsInstance(user, AuthenticatedUser)
        self.assertEqual(user.email, "dev@devlens.ai")
        self.assertTrue(user.user_id.startswith("usr_devlens_dev_"))

    def test_legacy_hackora_dev_token_accepted_in_dev_for_migration(self):
        user = get_current_user("Bearer hackora_dev_test123456789")
        self.assertIsInstance(user, AuthenticatedUser)
        self.assertEqual(user.email, "dev@devlens.ai")
        self.assertTrue(user.user_id.startswith("usr_hackora_dev_"))

    def test_demo_token_strictly_rejected_in_production(self):
        # Temporarily switch to production environment
        os.environ["ENVIRONMENT"] = "production"
        os.environ["DEBUG"] = "false"
        os.environ["ALLOW_DEV_TOKENS"] = "false"
        try:
            self.assertFalse(is_dev_environment())
            
            # DevLens demo token must be rejected in production
            with self.assertRaises(HTTPException) as ctx1:
                get_current_user("Bearer devlens_dev_malicious_attacker")
            self.assertEqual(ctx1.exception.status_code, 401)
            self.assertIn("restricted to permitted development environments", ctx1.exception.detail)

            # Legacy HACKORA demo token must also be rejected in production
            with self.assertRaises(HTTPException) as ctx2:
                get_current_user("Bearer hackora_dev_malicious_attacker")
            self.assertEqual(ctx2.exception.status_code, 401)
            self.assertIn("restricted to permitted development environments", ctx2.exception.detail)
        finally:
            os.environ["ENVIRONMENT"] = "development"
            os.environ["DEBUG"] = "true"

    def test_valid_jwt_claims_decoding(self):
        secret = "test_supabase_jwt_secret_32_chars_long!!"
        os.environ["SUPABASE_JWT_SECRET"] = secret
        try:
            token = jwt.encode(
                {"sub": "user_12345", "email": "engineer@company.com", "aud": "authenticated", "exp": time.time() + 3600},
                secret,
                algorithm="HS256"
            )
            user = get_current_user(f"Bearer {token}")
            self.assertEqual(user.user_id, "user_12345")
            self.assertEqual(user.email, "engineer@company.com")
        finally:
            os.environ.pop("SUPABASE_JWT_SECRET", None)

if __name__ == "__main__":
    unittest.main()

