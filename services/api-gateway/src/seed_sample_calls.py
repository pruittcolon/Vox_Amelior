#!/usr/bin/env python3
"""
Sample Call Data Generator for Call Intelligence Portal

Generates synthetic credit union call transcriptions for testing and demos.
Uses realistic scenarios based on common member inquiries.

Usage:
    python seed_sample_calls.py [--count N] [--db-path /path/to/calls.db]
"""

import argparse
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Any

# Add the src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from call_intelligence_manager import CallIntelligenceManager, get_call_intelligence_manager
except ImportError:
    print("Error: Could not import CallIntelligenceManager. Run from api-gateway/src directory.")
    sys.exit(1)

# Sample member IDs (SCU-style format)
SAMPLE_MEMBER_IDS = [
    "SCU-000123",
    "SCU-000456",
    "SCU-000789",
    "SCU-001234",
    "SCU-001567",
    "SCU-002345",
    "SCU-003456",
    "SCU-004567",
    "SCU-005678",
    "SCU-006789",
    "SCU-007890",
    "SCU-008901",
    "SCU-009012",
    "SCU-010123",
    "SCU-011234",
]

# Sample agent IDs
SAMPLE_AGENT_IDS = [
    "MSR-Sarah",
    "MSR-John",
    "MSR-Emily",
    "MSR-Michael",
    "MSR-Lisa",
    "LO-David",
    "LO-Jennifer",
    "FA-Robert",
    "FA-Michelle",
]

# Call transcript templates by scenario
CALL_SCENARIOS = [
    {
        "category": "account_access.login_issues",
        "intent": "Login Help",
        "duration_range": (120, 300),
        "sentiment_range": (-0.3, 0.2),
        "transcripts": [
            """MSR: Thank you for calling Service Credit Union, this is Sarah speaking. How may I assist you today?
MEMBER: Hi, I'm having trouble logging into my online banking. It keeps saying my password is wrong but I know I'm entering it correctly.
MSR: I'm sorry to hear that. Let me help you with that. Can you verify your member number please?
MEMBER: Sure, it's [ACCOUNT_REDACTED].
MSR: Thank you. I can see your account. It looks like there were multiple failed login attempts. For security, the account was temporarily locked.
MEMBER: Oh, I didn't realize that. What do I do now?
MSR: I can unlock it for you right now and send a password reset link to your registered email.
MEMBER: That would be great, thank you.
MSR: Done! You should receive the email within 5 minutes. Is there anything else I can help with?
MEMBER: No, that's all. Thank you so much for your help!""",
            """MSR: Service Credit Union, John speaking. What can I do for you?
MEMBER: I can't log in to the mobile app. It just shows an error.
MSR: I'll be happy to help. What error message are you seeing?
MEMBER: It says "Authentication Failed. Please try again."
MSR: Let me check your account status. Can I have your member number?
MEMBER: [ACCOUNT_REDACTED].
MSR: I see the issue. Your MFA settings were reset after the recent app update. Let me walk you through re-enabling it.
MEMBER: Okay, I appreciate that.""",
        ],
    },
    {
        "category": "transactions.missing_deposit",
        "intent": "Deposit Inquiry",
        "duration_range": (180, 420),
        "sentiment_range": (-0.5, -0.1),
        "transcripts": [
            """MSR: Thank you for calling Service Credit Union. This is Emily. How can I help?
MEMBER: Yes, I'm calling because I made a mobile deposit yesterday and it's still not showing in my account.
MSR: I understand that can be concerning. Let me look into this for you. May I have your member number?
MEMBER: It's [ACCOUNT_REDACTED].
MSR: Thank you. I can see the deposit you made. It was a check for $[AMOUNT_REDACTED] deposited yesterday at 4:45 PM.
MEMBER: Yes, that's the one. Where is it?
MSR: Mobile deposits made after 2 PM are processed the next business day. Since you deposited on Friday, it will be available Tuesday due to the holiday weekend.
MEMBER: Oh, I didn't realize that. I needed that money for rent.
MSR: I completely understand. I can offer you a courtesy advance of up to $500 if that would help. Would you like me to process that?
MEMBER: Yes please, that would be very helpful.
MSR: Done. The $500 is now available in your checking account. The remaining balance will be released Tuesday morning.
MEMBER: Thank you so much, you've been really helpful.""",
        ],
    },
    {
        "category": "cards.lost_stolen",
        "intent": "Lost Card Report",
        "duration_range": (180, 360),
        "sentiment_range": (-0.4, 0.0),
        "transcripts": [
            """MSR: Thank you for calling Service Credit Union. This is Michael. How may I help you today?
MEMBER: Hi, I think I lost my debit card. I can't find it anywhere and I'm worried someone might use it.
MSR: I'm sorry to hear that. Let me help you right away. First, can you confirm your identity? What's your member number?
MEMBER: [ACCOUNT_REDACTED].
MSR: Thank you. And can you confirm the last 4 digits of your Social Security?
MEMBER: [SSN_REDACTED].
MSR: Perfect, thank you. I'm blocking your current card right now... Done. No unauthorized transactions have been made.
MEMBER: Oh good, that's a relief.
MSR: I'll order you a new card right away. It will arrive in 5-7 business days. Would you like me to also set up a temporary digital card for immediate use?
MEMBER: Yes, that would be great!
MSR: I've added a digital card to your account. You can access it through our mobile app under "Cards" section. It's ready to use immediately.
MEMBER: Thank you so much for your help."""
        ],
    },
    {
        "category": "loans.payment_question",
        "intent": "Loan Payment Inquiry",
        "duration_range": (120, 240),
        "sentiment_range": (0.0, 0.4),
        "transcripts": [
            """LO: Thank you for calling Service Credit Union Loan Department. This is David. How can I assist you?
MEMBER: Hi, I wanted to check on my auto loan payment. I think I might have missed last month.
LO: Of course, let me pull up your account. May I have your member number?
MEMBER: [ACCOUNT_REDACTED].
LO: Thank you. I can see your auto loan here. Actually, your payment was received on the 15th. You're all current.
MEMBER: Oh really? That's great news! I was worried.
LO: No worries at all. Your next payment of $[AMOUNT_REDACTED] is due on February 15th.
MEMBER: Perfect. While I have you, what's my current payoff amount?
LO: Your current payoff balance is $[AMOUNT_REDACTED], good through the end of this month.
MEMBER: Great, thank you for all the info!
LO: You're welcome. Is there anything else I can help with today?
MEMBER: No, that's everything. Thanks!"""
        ],
    },
    {
        "category": "fraud.unauthorized_activity",
        "intent": "Fraud Report",
        "duration_range": (300, 600),
        "sentiment_range": (-0.6, -0.2),
        "transcripts": [
            """FA: Fraud Prevention Department, this is Michelle. How can I help you?
MEMBER: I just got a notification about a purchase I didn't make. I think someone is using my card!
FA: I understand this is concerning. Let me help you right away. Can you verify your identity first? Your member number please?
MEMBER: [ACCOUNT_REDACTED].
FA: Thank you. What was the notification about?
MEMBER: It said there was a charge for $[AMOUNT_REDACTED] at an electronics store in California. But I'm here in New Hampshire!
FA: I see that transaction. It's still pending. Let me block your card immediately to prevent any further charges.
MEMBER: Please do!
FA: Done. Card is blocked. I'm also flagging this as fraudulent and opening a dispute case for you. Case number is FRD-[PHONE_REDACTED].
MEMBER: What happens next?
FA: We'll investigate and issue a provisional credit within 24-48 hours. You won't be responsible for fraudulent charges. I'm also expediting a new card to you - it should arrive in 2-3 business days.
MEMBER: Thank you so much. This is really scary.
FA: I completely understand. You did the right thing calling us immediately. Is there anything else you'd like me to review on your account?
MEMBER: Can you check if there are any other suspicious transactions?
FA: Absolutely. Let me review your recent activity... I don't see any other unusual charges. The fraudster only got this one attempt through before we caught it.
MEMBER: Okay, thank you for your help."""
        ],
    },
    {
        "category": "digital_banking.app_issues",
        "intent": "Mobile App Support",
        "duration_range": (180, 360),
        "sentiment_range": (-0.2, 0.3),
        "transcripts": [
            """MSR: Service Credit Union, Lisa speaking. How can I help you today?
MEMBER: Hi, I just updated the mobile app and now it won't let me transfer money between my accounts.
MSR: I apologize for the inconvenience. Let me help you troubleshoot. What happens when you try to make a transfer?
MEMBER: I tap on transfer, select the accounts, enter the amount, but when I hit confirm it just shows a spinning wheel forever.
MSR: I see. There was an issue with the latest update affecting some devices. Can you tell me what phone you're using?
MEMBER: I have an iPhone 12.
MSR: Thank you. Try this: go to Settings, then the SCU app, and turn off "Background App Refresh" then turn it back on. Then force close the app and reopen it.
MEMBER: Okay, let me try... Okay, I did that. Let me try a transfer now... It's working! The transfer went through!
MSR: Excellent! I'm glad that resolved it. Our tech team is working on a fix for the next update.
MEMBER: Great, thanks for the help!"""
        ],
    },
    {
        "category": "fees.fee_dispute",
        "intent": "Fee Waiver Request",
        "duration_range": (180, 300),
        "sentiment_range": (-0.4, 0.1),
        "transcripts": [
            """MSR: Thank you for calling Service Credit Union. This is Sarah. How may I help you?
MEMBER: Hi, I'm calling about an overdraft fee on my account. I don't think it's fair because I had money transferred but it was just a few hours late.
MSR: I understand. Let me review your account. May I have your member number?
MEMBER: [ACCOUNT_REDACTED].
MSR: Thank you. I can see the overdraft occurred on the 3rd. Your scheduled transfer came in at 11 AM, but the debit processed at 8 AM.
MEMBER: Right, and I've been a member for 10 years. I've never had an overdraft before this.
MSR: I can see that, and we really appreciate your loyalty. Given your excellent account history, I'm happy to waive this fee for you as a one-time courtesy.
MEMBER: Really? Thank you so much!
MSR: You're welcome. I've credited the $35 back to your account. You should see it within an hour. Would you also like me to set up overdraft protection to prevent this in the future?
MEMBER: Yes, that would be great.
MSR: Done. I've linked your savings account as overdraft protection. If this happens again, we'll automatically transfer funds from savings to cover it.
MEMBER: Perfect, thank you so much for your help!"""
        ],
    },
    {
        "category": "member_services.address_change",
        "intent": "Account Update",
        "duration_range": (90, 180),
        "sentiment_range": (0.2, 0.5),
        "transcripts": [
            """MSR: Service Credit Union, how can I help you?
MEMBER: Hi, I just moved and need to update my address.
MSR: I'd be happy to help with that. For security, can you verify your member number and the last 4 of your Social?
MEMBER: Sure, member number is [ACCOUNT_REDACTED] and last 4 is [SSN_REDACTED].
MSR: Thank you. What's your new address?
MEMBER: It's [ADDRESS_REDACTED].
MSR: Got it. I've updated your address in our system. Your new cards and statements will go to this address.
MEMBER: Great! How long until I get a new card?
MSR: If your current card isn't expiring soon, you can keep using it. But I can order a new one with the updated address if you prefer.
MEMBER: No, that's okay. The current one is fine for now.
MSR: Perfect. Is there anything else?
MEMBER: Nope, that's everything. Thanks!"""
        ],
    },
]


def generate_call(scenario: dict[str, Any], days_ago: int = 0) -> dict[str, Any]:
    """Generate a single call record from a scenario template."""
    transcript = random.choice(scenario["transcripts"])
    duration = random.randint(*scenario["duration_range"])
    sentiment = random.uniform(*scenario["sentiment_range"])

    # Create random timestamp within the specified days
    call_time = datetime.utcnow() - timedelta(
        days=days_ago,
        hours=random.randint(8, 17),  # Business hours
        minutes=random.randint(0, 59),
    )

    return {
        "transcript": transcript,
        "member_id": random.choice(SAMPLE_MEMBER_IDS),
        "agent_id": random.choice(SAMPLE_AGENT_IDS),
        "duration_seconds": duration,
        "channel": "phone",
        "direction": random.choice(["inbound", "inbound", "inbound", "outbound"]),  # 75% inbound
        "fiserv_context": {"intent_category": scenario["intent"], "expected_sentiment": sentiment},
    }


def seed_calls(manager: CallIntelligenceManager, count: int = 50) -> int:
    """Seed the database with sample calls."""
    created = 0

    for i in range(count):
        # Pick a random scenario
        scenario = random.choice(CALL_SCENARIOS)

        # Generate call with random date in last 30 days
        days_ago = random.randint(0, 30)
        call_data = generate_call(scenario, days_ago)

        try:
            result = manager.ingest_call(**call_data)
            created += 1
            print(
                f"  Created call {result['call_id'][:8]}... (Member: {call_data['member_id']}, Category: {scenario['category']})"
            )
        except Exception as e:
            print(f"  Error creating call: {e}")

    return created


def main():
    parser = argparse.ArgumentParser(description="Seed sample call data for Call Intelligence Portal")
    parser.add_argument("--count", type=int, default=50, help="Number of sample calls to create")
    parser.add_argument(
        "--db-path", type=str, default=None, help="Path to calls.db (defaults to /app/instance/calls.db)"
    )
    args = parser.parse_args()

    print("🚀 Service Credit Union - Call Intelligence Data Seeder")
    print("=" * 60)

    # Initialize manager
    if args.db_path:
        manager = CallIntelligenceManager(db_path=args.db_path)
    else:
        manager = get_call_intelligence_manager()

    print(f"📁 Database: {manager.db_path}")
    print(f"📞 Creating {args.count} sample calls...")
    print()

    created = seed_calls(manager, args.count)

    print()
    print("=" * 60)
    print(f"✅ Created {created} sample calls")

    # Show stats
    stats = manager.get_dashboard_stats()
    print()
    print("📊 Dashboard Stats:")
    print(f"   Total Calls: {stats['total_calls']}")
    print(f"   Calls Today: {stats['calls_today']}")
    print(f"   Avg Sentiment: {stats['avg_sentiment']:.2f}")
    print(f"   Critical Issues (24h): {stats['critical_issues_24h']}")
    if stats.get("top_problem_this_week"):
        print(
            f"   Top Problem: {stats['top_problem_this_week']['name']} ({stats['top_problem_this_week']['count']} occurrences)"
        )


if __name__ == "__main__":
    main()
