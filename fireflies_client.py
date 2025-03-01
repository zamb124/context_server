import re

import httpx
import json
import asyncio
from config import config


class FirefliesAPI:
    def __init__(self):
        self.api_key = config.FIREFLIES_API_KEY
        self.base_url = "https://api.fireflies.ai/graphql"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    async def get_transcript(self, transcript_id: str) -> dict:
        """
        Retrieves a single transcript by its ID.

        Args:
            transcript_id: The ID of the transcript to retrieve.

        Returns:
            A dictionary containing the transcript data, or None if the transcript
            is not found or an error occurs.
        """
        pattern = r"(?:https:\/\/app\.fireflies\.ai\/view\/[\w-]+::)?([\w]+)$"

        match = re.match(pattern, transcript_id)
        if match:
            transcript_id = match.group(1)  # Вернет только id
        payload = {
            "query": """
                query Transcript($transcriptId: String!) {
                    transcript(id: $transcriptId) {
                        id
                        dateString
                        privacy
                        speakers {
                            id
                            name
                        }
                        sentences {
                            index
                            speaker_name
                            speaker_id
                            text
                            raw_text
                            start_time
                            end_time
                            ai_filters {
                                task
                                pricing
                                metric
                                question
                                date_and_time
                                text_cleanup
                                sentiment
                            }
                        }
                        title
                        host_email
                        organizer_email
                        calendar_id
                        user {
                            user_id
                            email
                            name
                            num_transcripts
                            recent_meeting
                            minutes_consumed
                            is_admin
                            integrations
                        }
                        fireflies_users
                        participants
                        date
                        transcript_url
                        audio_url
                        video_url
                        duration
                        meeting_attendees {
                            displayName
                            email
                            phoneNumber
                            name
                            location
                        }
                        summary {
                            keywords
                            action_items
                            outline
                            shorthand_bullet
                            overview
                            bullet_gist
                            gist
                            short_summary
                            short_overview
                            meeting_type
                            topics_discussed
                            transcript_chapters
                        }
                        cal_id
                        calendar_type
                        apps_preview {
                            outputs {
                                transcript_id
                                user_id
                                app_id
                                created_at
                                title
                                prompt
                                response
                            }
                        }
                        meeting_link
                    }
                }
            """,
            "variables": {"transcriptId": transcript_id}
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.base_url, headers=self.headers, json=payload, timeout=60)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                data = response.json()
                return data.get("data", {}).get("transcript")
            except httpx.HTTPStatusError as e:
                print(f"HTTP error: {e}")
                return None
            except httpx.RequestError as e:
                print(f"Request error: {e}")
                return None
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return None

    async def get_transcripts(self, title: str = None, date: float = None, limit: int = None, skip: int = None,
                              host_email: str = None, participant_email: str = None, user_id: str = None) -> list:
        """
        Retrieves a list of transcripts based on specified filters.

        Args:
            title: Filter by transcript title.
            date: Filter by transcript date (timestamp).
            limit: Limit the number of results returned.
            skip: Skip the first N results.
            host_email: Filter by host email.
            participant_email: Filter by participant email.
            user_id: Filter by user ID.

        Returns:
            A list of transcript dictionaries, or None if an error occurs.
        """

        payload = {
            "query": """
                query Transcripts(
                    $title: String
                    $date: Float
                    $limit: Int
                    $skip: Int
                    $hostEmail: String
                    $participantEmail: String
                    $userId: String
                ) {
                    transcripts(
                        title: $title
                        date: $date
                        limit: $limit
                        skip: $skip
                        host_email: $hostEmail
                        participant_email: $participantEmail
                        user_id: $userId
                    ) {
                        id
                        sentences {
                            index
                            speaker_name
                            speaker_id
                            text
                            raw_text
                            start_time
                            end_time
                            ai_filters {
                                task
                                pricing
                                metric
                                question
                                date_and_time
                                text_cleanup
                                sentiment
                            }
                        }
                        title
                        speakers {
                            id
                            name
                        }
                        host_email
                        organizer_email
                        meeting_info {
                            fred_joined
                            silent_meeting
                            summary_status
                        }
                        calendar_id
                        user {
                            user_id
                            email
                            name
                            num_transcripts
                            recent_meeting
                            minutes_consumed
                            is_admin
                            integrations
                        }
                        fireflies_users
                        participants
                        date
                        transcript_url
                        audio_url
                        video_url
                        duration
                        meeting_attendees {
                            displayName
                            email
                            phoneNumber
                            name
                            location
                        }
                        summary {
                            keywords
                            action_items
                            outline
                            shorthand_bullet
                            overview
                            bullet_gist
                            gist
                            short_summary
                            short_overview
                            meeting_type
                            topics_discussed
                            transcript_chapters
                        }
                        cal_id
                        calendar_type
                        apps_preview {
                            outputs {
                                transcript_id
                                user_id
                                app_id
                                created_at
                                title
                                prompt
                                response
                            }
                        }
                        meeting_link
                    }
                }
            """,
            "variables": {
                "title": title,
                "date": date,
                "limit": limit,
                "skip": skip,
                "hostEmail": host_email,
                "participantEmail": participant_email,
                "userId": user_id
            }
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(self.base_url, headers=self.headers, json=payload)
                response.raise_for_status()
                data = response.json()
                return data.get("data", {}).get("transcripts", [])
            except httpx.HTTPStatusError as e:
                print(f"HTTP error: {e}")
                return None
            except httpx.RequestError as e:
                print(f"Request error: {e}")
                return None
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                return None

    async def close(self):
        """Closes the httpx client session."""
        if hasattr(self, 'client') and self.client:
            await self.client.aclose()


fireflies_api = FirefliesAPI()