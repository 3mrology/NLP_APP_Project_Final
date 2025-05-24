import fix_pwd
import os
import json
import torch
import base64
import re
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from transformers import pipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.schema import Document

# Azure OpenAI integration
class AzureLLMProvider:
    def __init__(self, api_key=None, api_base=None, api_version=None, deployment_name=None, temperature=0.7):
        """Initialize Azure OpenAI provider"""
        self.api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self.api_base = api_base or os.environ.get("AZURE_OPENAI_API_BASE", "https://your-endpoint.openai.azure.com/")
        self.api_version = api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2023-12-01-preview")
        self.deployment_name = deployment_name or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
        self.temperature = temperature
        
        # Debug output - uncomment to see what's happening
        print(f"API Key length: {len(self.api_key) if self.api_key else 'None'}")
        print(f"API Base: {self.api_base}")
        print(f"Deployment: {self.deployment_name}")
        
        # Initialize the LLM
        try:
            from langchain_openai import AzureChatOpenAI
            self.llm = AzureChatOpenAI(
                openai_api_key=self.api_key,
                azure_endpoint=self.api_base,
                openai_api_version=self.api_version,
                deployment_name=self.deployment_name,
                temperature=self.temperature
            )
            # Test with a simple prediction
            test_result = self.llm.predict("Say 'API connection successful'")
            print(f"API Test Result: {test_result}")
        except Exception as e:
            print(f"❌ Error initializing Azure OpenAI: {str(e)}")
            self.llm = None

    
    
    def predict(self, text):
        """Generate a prediction using the LLM"""
        try:
            return self.llm.predict(text)
        except Exception as e:
            print(f"Error in LLM prediction: {str(e)}")
            return "I apologize, but I'm having trouble processing that. Could we try a different approach?"

# Improved Local Event Extractor (from paste-2.txt)
class ImprovedLocalExtractor:
    """A rule-based extractor that requires no API calls"""
    def __init__(self):
        # Time-related keywords to identify potential events
        self.time_indicators = [
            'childhood', 'youth', 'teenager', 'college', 'school', 'university',
            'year', 'month', 'week', 'day', 'morning', 'afternoon', 'evening', 'night',
            'birthday', 'holiday', 'vacation', 'trip', 'journey', 'weekend',
            'january', 'february', 'march', 'april', 'may', 'june', 'july', 
            'august', 'september', 'october', 'november', 'december',
            'winter', 'spring', 'summer', 'fall', 'autumn',
            '19', '20', # Likely part of years like 1990, 2010
            'when i was', 'back in', 'during', 'before', 'after', 'while',
            'growing up', 'younger', 'older', 'graduating',
            'first time', 'last time', 'remember'
        ]
        
        # People-related keywords
        self.people_indicators = [
            'mother', 'father', 'mom', 'dad', 'parent', 'parents',
            'brother', 'sister', 'sibling', 'siblings', 'family',
            'grandmother', 'grandfather', 'grandma', 'grandpa', 'grandparent', 
            'aunt', 'uncle', 'cousin', 'relative', 'relatives',
            'husband', 'wife', 'spouse', 'partner', 'boyfriend', 'girlfriend',
            'child', 'children', 'son', 'daughter', 'kid', 'kids',
            'friend', 'friends', 'colleague', 'colleagues', 'coworker', 'coworkers',
            'teacher', 'student', 'classmate', 'classmates', 'roommate', 'roommates',
            'boss', 'supervisor', 'employee', 'manager', 'director', 'leader'
        ]
        
        # Emotion-related keywords to identify significant events
        self.emotion_indicators = [
            'happy', 'sad', 'angry', 'excited', 'scared', 'afraid', 'worried',
            'nervous', 'anxious', 'calm', 'peaceful', 'relaxed', 'stressed',
            'surprised', 'shocked', 'amazed', 'astonished', 'overwhelmed',
            'proud', 'ashamed', 'embarrassed', 'guilty', 'jealous', 'envious',
            'lonely', 'loved', 'appreciated', 'grateful', 'thankful',
            'disappointed', 'frustrated', 'annoyed', 'irritated',
            'joyful', 'delighted', 'thrilled', 'elated', 'ecstatic',
            'depressed', 'miserable', 'gloomy', 'heartbroken'
        ]
        
        # Topic categories
        self.topic_categories = {
            'family': ['family', 'mother', 'father', 'brother', 'sister', 'parent', 'child'],
            'education': ['school', 'college', 'university', 'class', 'study', 'learn', 'teacher', 'student'],
            'career': ['job', 'work', 'career', 'profession', 'business', 'company', 'office'],
            'relationship': ['friend', 'girlfriend', 'boyfriend', 'spouse', 'partner', 'date', 'love', 'marriage'],
            'travel': ['travel', 'trip', 'vacation', 'journey', 'visit', 'tour', 'country', 'city'],
            'health': ['health', 'sick', 'illness', 'disease', 'hospital', 'doctor', 'medicine'],
            'achievement': ['achievement', 'success', 'accomplish', 'win', 'award', 'graduate', 'finish'],
            'loss': ['loss', 'lose', 'death', 'die', 'funeral', 'grief', 'mourn'],
            'celebration': ['celebration', 'party', 'birthday', 'wedding', 'anniversary', 'holiday', 'festival'],
            'hobby': ['hobby', 'sport', 'game', 'music', 'art', 'read', 'book', 'movie', 'play']
        }
    
    def extract_date(self, text):
        """Extract potential date information from text"""
        # Try to find years (19xx or 20xx)
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text)
        if year_match:
            return year_match.group(1)
            
        # Try to find age indicators
        age_match = re.search(r'when I was (\d+|a teenager|a child|a kid|young|in college|in school)', text, re.IGNORECASE)
        if age_match:
            return f"Age: {age_match.group(1)}"
            
        # Look for season/month + potential year
        season_month_match = re.search(r'\b(winter|spring|summer|fall|autumn|january|february|march|april|may|june|july|august|september|october|november|december)\s+(?:of\s+)?(19\d{2}|20\d{2})?\b', text, re.IGNORECASE)
        if season_month_match:
            if season_month_match.group(2):  # If year is captured
                return f"{season_month_match.group(1)} {season_month_match.group(2)}"
            else:
                return season_month_match.group(1).capitalize()
                
        # Look for relative time indicators
        for indicator in ['childhood', 'youth', 'teenage years', 'college years', 'early twenties', 'high school']:
            if indicator in text.lower():
                return indicator.capitalize()
                
        # Default unknown date with confidence level
        return "Unknown date"
    
    def extract_people(self, text):
        """Extract people mentioned in the text"""
        people = []
        text_lower = text.lower()
        
        # Find proper nouns (simplified approach - look for capitalized words)
        proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', text)
        for noun in proper_nouns:
            if len(noun) > 1 and noun.lower() not in ['i', 'me', 'my', 'mine', 'myself']:
                people.append(noun)
        
        # Look for relationship indicators
        for indicator in self.people_indicators:
            if indicator in text_lower:
                # Find the full phrase (e.g., "my mother" rather than just "mother")
                pattern = r'\b(?:my|our|their|his|her|the)\s+' + indicator + r'\b'
                matches = re.findall(pattern, text_lower)
                
                if matches:
                    for match in matches:
                        people.append(match.capitalize())
                elif indicator in text_lower:
                    people.append(indicator.capitalize())
        
        # Remove duplicates
        return list(set(people))
    
    def determine_topic(self, text):
        """Determine the most likely topic category for the text"""
        text_lower = text.lower()
        category_scores = {}
        
        # Count occurrences of each category's keywords
        for category, keywords in self.topic_categories.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    # Count occurrences
                    score += text_lower.count(keyword)
            category_scores[category] = score
        
        # Get the highest scoring category
        if not category_scores or max(category_scores.values()) == 0:
            return "Memory"  # Default topic
            
        max_category = max(category_scores, key=category_scores.get)
        return max_category.capitalize()
    
    def extract_events_from_text(self, text):
        """Extract potential events from a piece of text"""
        events = []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if sentence has time indicators
            has_time_indicator = any(indicator in sentence.lower() for indicator in self.time_indicators)
            
            # Check if sentence has people indicators
            has_people_indicator = any(indicator in sentence.lower() for indicator in self.people_indicators)
            
            # Check if sentence has emotion indicators
            has_emotion_indicator = any(indicator in sentence.lower() for indicator in self.emotion_indicators)
            
            # Consider it a potential event if it has time or people indicators
            is_potential_event = has_time_indicator or has_people_indicator or has_emotion_indicator
            
            if is_potential_event:
                date = self.extract_date(sentence)
                people = self.extract_people(sentence)
                topic = self.determine_topic(sentence)
                
                events.append({
                    "date": date,
                    "topic": topic,
                    "people": people,
                    "description": sentence
                })
        
        return events
    
    def extract_events_from_conversation(self, conversation):
        """Extract events from a conversation"""
        events = []
        
        # Split by speaker
        lines = conversation.strip().split('\n')
        
        # Extract patient/user lines
        patient_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('Patient:') or line.startswith('User:'):
                patient_text = line.split(':', 1)[1].strip()
                patient_lines.append(patient_text)
        
        # Process each patient line
        for line in patient_lines:
            line_events = self.extract_events_from_text(line)
            events.extend(line_events)
        
        # If no events found, try to extract from the entire conversation
        if not events:
            conversation_text = ' '.join(patient_lines)
            events = self.extract_events_from_text(conversation_text)
        
        # If still no events, create some placeholder events
        if not events:
            events = self.create_placeholder_events(patient_lines)
            
        return events
    
    def create_placeholder_events(self, patient_lines):
        """Create placeholder events when no events are found"""
        events = []
        
        # Create a general memory if we have any text
        if patient_lines:
            full_text = ' '.join(patient_lines)
            
            # Create a general memory
            events.append({
                "date": "Recent memory",
                "topic": "General",
                "people": self.extract_people(full_text),
                "description": "A general memory mentioned during conversation"
            })
            
            # Create common life stage memories
            life_stages = ["Childhood", "Teenage years", "Young adulthood", "Recent past"]
            
            for stage in life_stages:
                events.append({
                    "date": stage,
                    "topic": "Life stage",
                    "people": [],
                    "description": f"Memories from {stage.lower()}"
                })
                
        return events

# Memory Graph System (adapted from paste-2.txt)
class MemoryGraph:
    def __init__(self, use_api=False, llm=None):
        """Initialize memory graph system"""
        self.use_api = use_api
        self.llm = llm
        self.graph = nx.DiGraph()
        self.local_extractor = ImprovedLocalExtractor()
        
    def extract_events_from_conversation(self, conversation):
        """Extract events from conversation, using only local methods by default"""
        print("Extracting events from conversation...")
        
        if self.use_api and self.llm:
            try:
                print("Using API for event extraction...")
                extract_prompt = PromptTemplate.from_template(
                    """You are given a conversation between a counselor and a user:
                    ====== Conversation Begin ======
                    {conversation}
                    ====== Conversation End ======
                    
                    Read the conversation carefully and list all the events/moments/stories/experiences alone or
                    with others mentioned by the patient in detail and the date these events happened. Please list
                    as many as possible. Your output should be in the following format:
                    1. <date>#<topic>#<people-involved>#<description in detail>
                    2. <date>#<topic>#<people-involved>#<description in detail>
                    ...
                    
                    e.g.,
                    1. 1980 early#Birthday Party#Michelle, Adolf, neighbors#<descriptions of this party in detail>
                    
                    These events should be ranked in chronological order.
                    """
                )
                
                prompt = extract_prompt.format(conversation=conversation)
                response = self.llm.predict(text=prompt)
                
                events = []
                for line in response.strip().split('\n'):
                    if line and line[0].isdigit() and '#' in line:
                        event_text = line.split('. ', 1)[1] if '. ' in line else line
                        try:
                            date, topic, people, description = event_text.split('#', 3)
                            events.append({
                                "date": date.strip(),
                                "topic": topic.strip(),
                                "people": [p.strip() for p in people.split(',')],
                                "description": description.strip()
                            })
                        except ValueError:
                            continue
                            
                if events:
                    return events
                    
                print("No events found by API, falling back to local extraction...")
                
            except Exception as e:
                print(f"Error with API event extraction: {str(e)}")
                print("Falling back to local extraction...")
        
        # Use local extraction as primary or fallback method
        return self.local_extractor.extract_events_from_conversation(conversation)
    
    def add_events_to_graph(self, events):
        """Add extracted events to the memory graph"""
        for i, event in enumerate(events):
            event_id = f"event_{len(self.graph.nodes)}"
            
            # Add node with all properties
            self.graph.add_node(
                event_id,
                date=event["date"],
                topic=event["topic"],
                people=event["people"],
                description=event["description"]
            )
            
            # Connect events with people in common
            for existing_id in list(self.graph.nodes()):
                if existing_id == event_id:
                    continue
                    
                existing_people = self.graph.nodes[existing_id].get("people", [])
                
                # Check for common people
                common_people = set(event["people"]) & set(existing_people)
                if common_people:
                    self.graph.add_edge(event_id, existing_id, relationship="common_people", people=list(common_people))
                
                # Check for temporal relationship (very basic)
                try:
                    existing_date = self.graph.nodes[existing_id].get("date", "")
                    current_date = event["date"]
                    
                    # Try to determine order between life stages
                    life_stages = ["Childhood", "Teenage years", "Young adulthood", "Recent past"]
                    
                    if existing_date in life_stages and current_date in life_stages:
                        existing_index = life_stages.index(existing_date)
                        current_index = life_stages.index(current_date)
                        
                        if existing_index < current_index:
                            self.graph.add_edge(existing_id, event_id, relationship="happened_before")
                        elif existing_index > current_index:
                            self.graph.add_edge(event_id, existing_id, relationship="happened_before")
                    # For actual years, compare numerically
                    elif existing_date.isdigit() and current_date.isdigit():
                        if int(existing_date) < int(current_date):
                            self.graph.add_edge(existing_id, event_id, relationship="happened_before")
                        elif int(existing_date) > int(current_date):
                            self.graph.add_edge(event_id, existing_id, relationship="happened_before")
                except:
                    # Skip date comparison if there's an error
                    pass
                
        return self.graph
    
    def generate_follow_up_questions(self, max_questions=3):
        """Generate follow-up questions based on the memory graph
        Using a template approach or API if available"""
        # Use API-based approach if available
        if self.use_api and self.llm and self.graph.nodes:
            try:
                # Extract all events from the memory graph
                all_events = []
                for node_id, data in self.graph.nodes(data=True):
                    all_events.append({
                        "date": data.get("date", "Unknown"),
                        "topic": data.get("topic", "Unknown"),
                        "people": data.get("people", []),
                        "description": data.get("description", "")
                    })
                
                # Format events for the prompt
                events_text = ""
                for i, event in enumerate(all_events):
                    events_text += f"{i+1}. Date: {event['date']}, Topic: {event['topic']}, "
                    events_text += f"People: {', '.join(event['people'])}, "
                    events_text += f"Description: {event['description']}\n"
                
                # Generate questions using LLM
                prompt = f"""
                Based on the following memory events extracted from a patient's conversation:
                
                {events_text}
                
                Generate {max_questions} follow-up questions that would be therapeutic and help the patient:
                1. Explore these memories in more depth
                2. Connect these memories to their current well-being
                3. Discover positive aspects or growth from these experiences
                
                Format your response as a numbered list of questions only.
                """
                
                response = self.llm.predict(text=prompt)
                
                # Extract questions from response
                questions = []
                for line in response.strip().split('\n'):
                    line = line.strip()
                    if line and (line[0].isdigit() or line.startswith('-')):
                        question_text = re.sub(r'^\d+\.\s*|-\s*', '', line).strip()
                        if question_text and question_text[-1] not in '.?!':
                            question_text += '?'
                        if question_text:
                            questions.append(question_text)
                
                return questions[:max_questions]
                
            except Exception as e:
                print(f"Error generating questions with API: {str(e)}")
                print("Falling back to template-based questions...")
        
        # Template-based approach as fallback
        if not self.graph.nodes:
            # Default questions when graph is empty
            return [
                "Could you tell me about a positive memory from your childhood?",
                "Who is someone who has had a significant impact on your life?",
                "What is an achievement you're particularly proud of?"
            ]
        
        questions = []
        
        # 1. Get unique topics in the graph
        topics = set()
        dates = set()
        people = set()
        
        for _, data in self.graph.nodes(data=True):
            if "topic" in data:
                topics.add(data["topic"])
            if "date" in data:
                dates.add(data["date"])
            if "people" in data:
                for person in data.get("people", []):
                    people.add(person)
        
        # 2. Template questions based on graph content
        
        # Date-based questions
        date_templates = [
            "Can you tell me more about your experiences during {date}?",
            "What other memories do you have from {date}?",
            "How did your experiences in {date} shape who you are today?",
            "What were the most important lessons you learned during {date}?"
        ]
        
        # Topic-based questions
        topic_templates = [
            "You mentioned {topic}. Could you share another experience related to this?",
            "How did {topic} influence your perspectives or values?",
            "What other significant {topic} experiences have you had?",
            "How do you feel when you think back on {topic} events in your life?"
        ]
        
        # People-based questions
        people_templates = [
            "You mentioned {person}. How has your relationship with them evolved over time?",
            "What is another memorable experience you've had with {person}?",
            "How has {person} influenced your life or shaped who you are?",
            "What qualities do you admire most about {person}?"
        ]
        
        # General follow-up questions
        general_templates = [
            "How did that experience affect you emotionally?",
            "What did you learn from that experience?",
            "How has that memory influenced your decisions or actions since then?",
            "Looking back, how do you feel about that experience now?",
            "Were there other significant moments from that period you'd like to share?"
        ]
        
        # 3. Generate questions using templates
        import random
        
        # Add date-based questions
        if dates:
            date = random.choice(list(dates))
            template = random.choice(date_templates)
            questions.append(template.format(date=date))
        
        # Add topic-based questions
        if topics:
            topic = random.choice(list(topics))
            template = random.choice(topic_templates)
            questions.append(template.format(topic=topic.lower()))
        
        # Add people-based questions
        if people:
            person = random.choice(list(people))
            template = random.choice(people_templates)
            questions.append(template.format(person=person))
        
        # Fill remaining slots with general questions
        while len(questions) < max_questions:
            template = random.choice(general_templates)
            if template not in questions:
                questions.append(template)
        
        # Shuffle and limit to max_questions
        random.shuffle(questions)
        return questions[:max_questions]
    
    def visualize_graph(self, output_file="memory_graph.png"):
        """Visualize the memory graph"""
        if not self.graph.nodes:
            print("Graph is empty, nothing to visualize")
            return None
            
        plt.figure(figsize=(12, 8))
        
        # Create position layout
        pos = nx.spring_layout(self.graph, seed=42)  # Fixed seed for reproducibility
        
        # Draw nodes
        node_colors = []
        for node in self.graph.nodes():
            topic = self.graph.nodes[node].get("topic", "").lower()
            if "family" in topic:
                node_colors.append("lightblue")
            elif "education" in topic:
                node_colors.append("lightgreen")
            elif "career" in topic:
                node_colors.append("orange")
            elif "relationship" in topic:
                node_colors.append("pink")
            else:
                node_colors.append("lightgray")
                
        nx.draw_networkx_nodes(self.graph, pos, node_size=700, node_color=node_colors, alpha=0.8)
        
        # Draw edges
        edge_colors = []
        for _, _, data in self.graph.edges(data=True):
            relationship = data.get("relationship", "")
            if relationship == "happened_before":
                edge_colors.append("blue")
            elif relationship == "common_people":
                edge_colors.append("green")
            else:
                edge_colors.append("gray")
                
        nx.draw_networkx_edges(self.graph, pos, width=1.5, edge_color=edge_colors, alpha=0.7)
        
        # Draw labels
        labels = {}
        for node, data in self.graph.nodes(data=True):
            date = data.get("date", "")
            topic = data.get("topic", "")
            labels[node] = f"{topic}\n({date})"
            
        nx.draw_networkx_labels(self.graph, pos, labels=labels, font_size=8, font_weight="bold")
        
        # Add title and legend
        plt.title("Memory Graph: Events and Relationships", fontsize=16)
        
        # Create a custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=10, label='Family'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=10, label='Education'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Career'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='pink', markersize=10, label='Relationship'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=10, label='Other'),
            Line2D([0], [0], color='blue', lw=2, label='Temporal Relationship'),
            Line2D([0], [0], color='green', lw=2, label='Common People')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.axis("off")
        plt.tight_layout()
        
        # Save to buffer for returning
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        
        # Also save to file if requested
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            
        plt.close()
        
        print(f"Graph visualization saved to {output_file}")
        return buf
        
    def save_graph(self, file_path="memory_graph.json"):
        """Save the memory graph to a file"""
        # Convert to node-link format for JSON serialization
        data = nx.node_link_data(self.graph, link="links")
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Graph saved to {file_path}")
            
    def load_graph(self, file_path="memory_graph.json"):
        """Load the memory graph from a file"""
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            # Convert from node-link format
            self.graph = nx.node_link_graph(data, link="links")
            print(f"Graph loaded from {file_path}")
            return True
        return False
    
    def get_relevant_memories(self, query, max_memories=3):
        """Get memories relevant to a query"""
        if not self.graph.nodes or not query:
            return []
            
        # If API available, use semantic similarity
        if self.use_api and self.llm:
            try:
                # Extract all events from the memory graph
                all_events = []
                for node_id, data in self.graph.nodes(data=True):
                    all_events.append({
                        "id": node_id,
                        "date": data.get("date", "Unknown"),
                        "topic": data.get("topic", "Unknown"),
                        "people": data.get("people", []),
                        "description": data.get("description", "")
                    })
                
                # Format events for the prompt
                events_text = ""
                for i, event in enumerate(all_events):
                    events_text += f"{i+1}. Date: {event['date']}, Topic: {event['topic']}, "
                    events_text += f"People: {', '.join(event['people'])}, "
                    events_text += f"Description: {event['description']}\n"
                
                # Find relevant memories using LLM
                prompt = f"""
                Below are memories from a patient's life:
                
                {events_text}
                
                Given the patient's current statement/question: "{query}"
                
                List the indices of the {max_memories} most relevant memories from above that would help respond to 
                the patient's statement. Return only the numbers separated by commas, e.g., "2, 5, 7".
                """
                
                response = self.llm.predict(text=prompt)
                
                # Extract indices from response
                indices = []
                for match in re.finditer(r'\b\d+\b', response):
                    index = int(match.group()) - 1  # Convert to 0-based index
                    if 0 <= index < len(all_events):
                        indices.append(index)
                
                # Get the relevant memories
                relevant_memories = [all_events[i] for i in indices[:max_memories]]
                return relevant_memories
                
            except Exception as e:
                print(f"Error finding relevant memories with API: {str(e)}")
                print("Falling back to keyword matching...")
        
        # Fallback: Simple keyword matching
        relevant_memories = []
        query_lower = query.lower()
        
        for node_id, data in self.graph.nodes(data=True):
            score = 0
            
            # Check topic match
            topic = data.get("topic", "").lower()
            if topic in query_lower:
                score += 3
                
            # Check for people mentions
            for person in data.get("people", []):
                if person.lower() in query_lower:
                    score += 2
                    
            # Check for date mentions
            date = data.get("date", "").lower()
            if date in query_lower:
                score += 2
                
            # Check for word overlap in description
            description = data.get("description", "").lower()
            for word in query_lower.split():
                if word in description:
                    score += 0.5
                    
            if score > 0:
                relevant_memories.append({
                    "id": node_id,
                    "date": data.get("date", "Unknown"),
                    "topic": data.get("topic", "Unknown"),
                    "people": data.get("people", []),
                    "description": data.get("description", ""),
                    "relevance": score
                })
                
        # Sort by relevance score and return top matches
        relevant_memories.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        return relevant_memories[:max_memories]

# Empathetic Engagement module (adapted from paste-3.txt)
class EmpatheticEngagement:
    def __init__(self, llm=None):
        """Initialize the empathetic engagement system with emotion detection and therapeutic strategies"""
        self.llm = llm
        
        # Initialize emotion detection model
        try:
            self.emotion_classifier = pipeline(
                "text-classification", 
                model="j-hartmann/emotion-english-distilroberta-base",
                return_all_scores=True
            )
            self.emotion_detection_available = True
        except Exception as e:
            print(f"Warning: Could not load emotion detection model: {str(e)}")
            print("Using simplified emotion detection.")
            self.emotion_detection_available = False
        
        # Core therapeutic strategies
        self.therapy_strategies = {
            "reflective_listening": self._generate_reflective_listening_prompt,
            "cbt": self._generate_cbt_prompt,
            "psychodynamic": self._generate_psychodynamic_prompt,
            "general": self._generate_general_therapy_prompt
        }


    def generate_response_minimal(self, user_input, conversation_history, memory_nodes=None, character_persona=None):
        """Generate responses without using the API"""
        # Detect emotion locally
        emotion, intensity = self.detect_emotion(user_input)
        
        # Get the last few messages for context
        recent_messages = conversation_history.split('\n')[-6:]
        recent_text = '\n'.join(recent_messages)
        
        # Check for simple yes/no responses
        if user_input.lower() in ["yes", "yeah", "yep", "sure", "ok", "okay"]:
            if "diagnostics" in recent_text.lower() or "causing" in recent_text.lower():
                return "Tell me more about when this feeling started. What was happening in your life at that time?"
            else:
                return "Great. Could you share more details about what's been going on?"
        
        if user_input.lower() in ["no", "nope", "nah", "not really"]:
            return "That's okay. Maybe we could explore another aspect of what you're feeling. What would you say is most affecting your mood right now?"
        
        # Check for short answers
        if len(user_input.split()) < 5:
            return "I'd like to understand more. Could you elaborate on that?"
        
        # Prepare character-specific responses
        iron_man_responses = [
            f"I'm picking up that you're feeling {emotion}. Not my usual territory, but I'm all ears. What triggered this?",
            "Even genius billionaires have feelings. Walk me through what's going on in that head of yours.",
            "Let's debug this situation. When did you first notice feeling this way?",
            "Sometimes you gotta run before you can walk. Tell me more about what's causing this.",
            "My emotional sensors are detecting some {emotion}. What's the source code behind that?"
        ]
        
        captain_america_responses = [
            f"I understand feeling {emotion} can be difficult. Tell me more about what you're going through.",
            "In my experience, talking through our challenges helps us face them. What's weighing on you?",
            "The strongest people aren't afraid to admit when they're struggling. What's been happening?",
            "I've learned that resilience comes from acknowledging our feelings. Could you share more about yours?",
            "Even the toughest battles are easier when we don't face them alone. What's troubling you?"
        ]
        
        # Generic therapeutic responses
        generic_responses = [
            f"I notice you're feeling {emotion}. Can you tell me more about the circumstances around these feelings?",
            "What thoughts have been going through your mind lately?",
            "How have these feelings been affecting your daily life?",
            "When did you first start experiencing these feelings?",
            "Are there specific situations that tend to trigger these emotions?"
        ]
        
        # Select response based on character persona
        import random
        if character_persona and "iron man" in character_persona.lower():
            return random.choice(iron_man_responses)
        elif character_persona and "captain" in character_persona.lower():
            return random.choice(captain_america_responses)
        else:
            return random.choice(generic_responses)    
    
    def detect_emotion(self, text):
        """Detect emotion in text and determine intensity
        Returns emotion label and intensity score (0-1)"""
        if not self.emotion_detection_available:
            # Simple keyword-based backup
            emotions = {
                "joy": ["happy", "joy", "excited", "glad", "pleased", "delighted"],
                "sadness": ["sad", "depressed", "unhappy", "upset", "miserable"],
                "anger": ["angry", "mad", "furious", "irritated", "annoyed"],
                "fear": ["afraid", "scared", "worried", "anxious", "nervous"],
                "surprise": ["surprised", "shocked", "amazed", "astonished"],
                "disgust": ["disgusted", "revolted", "repulsed"],
                "neutral": ["okay", "fine", "alright", "neutral"]
            }
            
            text_lower = text.lower()
            max_emotion = "neutral"
            max_count = 0
            
            for emotion, keywords in emotions.items():
                count = sum(1 for keyword in keywords if keyword in text_lower)
                if count > max_count:
                    max_count = count
                    max_emotion = emotion
            
            # Calculate a simple intensity
            intensity = min(1.0, max_count * 0.2) if max_count > 0 else 0.5
            return max_emotion, intensity
            
        try:
            results = self.emotion_classifier(text)
            emotions = results[0]
            
            # Find the highest scoring emotion
            top_emotion = max(emotions, key=lambda x: x['score'])
            
            # Map model's emotions to our simplified set
            emotion_mapping = {
                "joy": "joy",
                "surprise": "surprise",
                "neutral": "neutral",
                "sadness": "sadness",
                "fear": "fear",
                "anger": "anger",
                "disgust": "disgust",
                "love": "joy",
                "admiration": "joy",
                "approval": "joy",
                "caring": "joy",
                "excitement": "joy",
                "gratitude": "joy",
                "pride": "joy",
                "optimism": "joy",
                "relief": "joy",
                "desire": "anticipation",
                "curiosity": "anticipation"
            }
            
            emotion = emotion_mapping.get(top_emotion['label'].lower(), top_emotion['label'].lower())
            intensity = top_emotion['score']
            
            return emotion, intensity
        except Exception as e:
            print(f"Error in emotion detection: {str(e)}")
            return "neutral", 0.5
    
    def determine_best_therapy_strategy(self, emotion, intensity):
        """Determine the best therapy strategy based on detected emotion and intensity"""
        # Simplified decision logic - in a real system this would be more sophisticated
        if emotion in ["sadness", "fear"] and intensity > 0.7:
            return "reflective_listening"
        elif emotion in ["anger", "disgust"] and intensity > 0.6:
            return "cbt"
        elif emotion in ["neutral", "joy", "surprise"] and intensity > 0.5:
            return "psychodynamic"
        else:
            return "general"
    
    def generate_response(self, user_input, conversation_history, 
                      memory_nodes=None, character_persona=None, session_context=None):
        """Generate a therapeutic response considering emotional state and appropriate strategies"""
        # Detect emotion
        emotion, intensity = self.detect_emotion(user_input)
        
        # Determine best strategy
        strategy = self.determine_best_therapy_strategy(emotion, intensity)
        
        # Generate strategy-specific prompt
        prompt_generator = self.therapy_strategies[strategy]
        prompt = prompt_generator(user_input, conversation_history, emotion, intensity, 
                                memory_nodes, character_persona, session_context)
        
        # Generate response with LLM if available
        if self.llm:
            try:
                response = self.llm.predict(text=prompt)
                return response
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                # Create a simple therapeutic response based on emotion
                if emotion == "sadness":
                    return "I hear that you're feeling sad. Would you like to talk about what's contributing to that feeling?"
                elif emotion == "fear" or emotion == "anxiety":
                    return "I can tell this is causing you some concern. Remember that we're in a safe space to explore these feelings."
                else:
                    return "I appreciate you sharing that with me. Could you tell me more about what's on your mind?"
        else:
            # Create a simple character-appropriate response if no LLM
            if character_persona and "iron man" in character_persona.lower():
                return f"Look, I'm picking up that you're feeling {emotion}. Not exactly my comfort zone, but I'm here. Want to run some diagnostics on what's causing that?"
            elif character_persona and "captain" in character_persona.lower():
                return f"I understand that feeling {emotion} can be challenging. In my experience, talking through it helps. What's weighing on your mind?"
            else:
                return f"I can see you're feeling {emotion}. Would you like to explore that together?"
    
    def _generate_reflective_listening_prompt(self, user_input, conversation_history, emotion, intensity,
                                             memory_nodes=None, character_persona=None, session_context=None):
        """Generate a prompt for reflective listening strategy"""
        memory_context = self._format_memory_context(memory_nodes)
        persona_context = self._format_persona_context(character_persona)
        session_info = self._format_session_context(session_context)
        
        prompt = f"""
        {persona_context}
        
        {session_info}
        
        The patient has expressed emotion: {emotion} with intensity: {intensity:.2f}.
        
        {memory_context}
        
        Please respond using reflective listening techniques:
        1. Acknowledge and validate their emotions
        2. Paraphrase their message to show understanding
        3. Focus on emotional undertones
        4. Use phrases like "I hear you saying that..." or "It sounds like you feel..."
        5. Don't rush to problem-solving
        
        Previous conversation:
        {conversation_history}
        
        Patient's message: {user_input}
        
        Your reflective response should make the patient feel truly understood:
        """
        
        return prompt
    
    def _generate_cbt_prompt(self, user_input, conversation_history, emotion, intensity,
                            memory_nodes=None, character_persona=None, session_context=None):
        """Generate a prompt for Cognitive-Behavioral Therapy strategy"""
        memory_context = self._format_memory_context(memory_nodes)
        persona_context = self._format_persona_context(character_persona)
        session_info = self._format_session_context(session_context)
        
        prompt = f"""
        {persona_context}
        
        {session_info}
        
        The patient has expressed emotion: {emotion} with intensity: {intensity:.2f}.
        
        {memory_context}
        
        Please respond using Cognitive-Behavioral Therapy techniques:
        1. Identify potential cognitive distortions in their thinking
        2. Gently challenge unhelpful thought patterns
        3. Encourage evidence-based thinking
        4. Suggest alternative perspectives
        5. Focus on actionable steps for positive change
        
        Previous conversation:
        {conversation_history}
        
        Patient's message: {user_input}
        
        Your CBT-informed response:
        """
        
        return prompt
    
    def _generate_psychodynamic_prompt(self, user_input, conversation_history, emotion, intensity,
                                      memory_nodes=None, character_persona=None, session_context=None):
        """Generate a prompt for Psychodynamic Therapy strategy"""
        memory_context = self._format_memory_context(memory_nodes)
        persona_context = self._format_persona_context(character_persona)
        session_info = self._format_session_context(session_context)
        
        prompt = f"""
        {persona_context}
        
        {session_info}
        
        The patient has expressed emotion: {emotion} with intensity: {intensity:.2f}.
        
        {memory_context}
        
        Please respond using Psychodynamic Therapy principles:
        1. Explore potential connections to past experiences
        2. Consider how past relationships might influence current feelings
        3. Look for patterns in their experiences
        4. Help uncover deeper meanings behind their thoughts and feelings
        5. Encourage self-reflection and insight
        
        Previous conversation:
        {conversation_history}
        
        Patient's message: {user_input}
        
        Your psychodynamic-informed response:
        """
        
        return prompt
    
    def _generate_general_therapy_prompt(self, user_input, conversation_history, emotion, intensity,
                                        memory_nodes=None, character_persona=None, session_context=None):
        """Generate a prompt for general therapeutic response"""
        memory_context = self._format_memory_context(memory_nodes)
        persona_context = self._format_persona_context(character_persona)
        session_info = self._format_session_context(session_context)
        
        prompt = f"""
        {persona_context}
        
        {session_info}
        
        The patient has expressed emotion: {emotion} with intensity: {intensity:.2f}.
        
        {memory_context}
        
        Please respond as a supportive therapist specializing in reminiscence therapy:
        1. Show empathy and genuine interest
        2. Encourage exploration of memories
        3. Ask thoughtful follow-up questions
        4. Connect past experiences to present well-being
        5. Maintain a warm, supportive tone
        
        Previous conversation:
        {conversation_history}
        
        Patient's message: {user_input}
        
        Your therapeutic response:
        """
        
        return prompt
    
    def _format_memory_context(self, memory_nodes):
        """Format memory nodes for inclusion in the prompt"""
        if not memory_nodes:
            return ""
            
        context = "Based on the patient's history, these memories appear relevant:\n"
        for i, node in enumerate(memory_nodes):
            date = node.get("date", "Unknown date")
            topic = node.get("topic", "General memory")
            people = ", ".join(node.get("people", []))
            description = node.get("description", "")
            
            context += f"{i+1}. {date}: {topic}"
            if people:
                context += f" involving {people}"
            context += f". {description}\n"
            
        return context
    
    def _format_persona_context(self, character_persona):
        """Format character persona instruction for inclusion in the prompt"""
        if not character_persona:
            return "You are a skilled therapist specializing in reminiscence therapy."
            
        persona_traits = {
            "iron_man": "witty, confident, intelligent, uses technical terms, makes pop culture references, occasionally sarcastic but ultimately supportive and caring beneath a tough exterior",
            "captain_america": "honest, principled, optimistic, uses straightforward language, respectful, occasionally old-fashioned phrases",
            "spider_man": "friendly, relatable, uses casual language, makes jokes to lighten the mood, enthusiastic, empathetic",
            "wonder_woman": "compassionate, wise, encouraging, uses empowering language, direct but supportive",
            "batman": "analytical, strategic, serious, speaks directly, asks probing questions, shows empathy through actions more than words",
            "thor": "confident, enthusiastic, occasionally formal speech patterns, good-natured, uses metaphors and colorful expressions"
        }
        
        # Get traits for the character, default to a general description if not found
        traits = persona_traits.get(
            character_persona.lower().replace(" ", "_"), 
            "unique, engaging, supportive while maintaining therapeutic professionalism"
        )
        
        return f"""You are a skilled therapist specializing in reminiscence therapy, taking on the persona of {character_persona}.
        Embody the personality traits of {character_persona}: {traits}.
        Your responses should sound like {character_persona} while maintaining therapeutic professionalism and effectiveness."""
    
    def _format_session_context(self, session_context):
        """Format session context for inclusion in the prompt"""
        if not session_context:
            return ""
            
        context = "Current therapy session context:\n"
        for key, value in session_context.items():
            context += f"- {key}: {value}\n"
            
        return context

# Document Processing System for RAG
class DocumentProcessor:
    def __init__(self, use_embeddings=True, llm=None):
        """Initialize document processor for RAG"""
        self.llm = llm
        self.use_embeddings = use_embeddings
        
        # Initialize embeddings if requested
        if use_embeddings:
            try:
                self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                print("Embeddings model loaded successfully.")
            except Exception as e:
                print(f"Error loading embeddings model: {str(e)}")
                print("Using keyword-based retrieval instead.")
                self.use_embeddings = False
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.db = None
        
        # Store extracted information
        self.documents = []
        self.summaries = {}
        self.people_mentioned = set()
        self.dates_mentioned = set()
        self.locations_mentioned = set()
    
    def process_document(self, file_path):
        """Process a document and add it to the vector store"""
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False
            
        try:
            # Load the document based on file type
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
                
            documents = loader.load()
            texts = self.text_splitter.split_documents(documents)
            
            # Store the documents
            self.documents.extend(texts)
            
            # Extract meaningful information
            self._extract_information(texts)
            
            # Create vector store if using embeddings
            if self.use_embeddings:
                if self.db is None:
                    self.db = FAISS.from_documents(texts, self.embeddings)
                else:
                    self.db.add_documents(texts)
                    
            print(f"Processed document: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error processing document {file_path}: {str(e)}")
            return False
    
    def _extract_information(self, documents):
        """Extract meaningful information from documents"""
        # Extract named entities using regex (simplified approach)
        person_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'
        date_pattern = r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b'
        location_pattern = r'\bin ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b'
        
        for doc in documents:
            text = doc.page_content
            
            # Find people
            people = re.findall(person_pattern, text)
            self.people_mentioned.update(people)
            
            # Find dates
            dates = re.findall(date_pattern, text)
            self.dates_mentioned.update(dates)
            
            # Find locations
            locations = re.findall(location_pattern, text)
            self.locations_mentioned.update(locations)
            
        # Create summaries using LLM if available
        if self.llm and len(documents) > 0:
            try:
                # Get the content of all documents
                all_text = "\n\n".join([doc.page_content for doc in documents])
                
                # Generate a summary
                prompt = f"""
                Please summarize the following document content in 2-3 paragraphs:
                
                {all_text[:4000]}  # Limit to first 4000 chars
                
                Focus on key events, people, and important personal information.
                """
                
                summary = self.llm.predict(text=prompt)
                self.summaries["main"] = summary
                
            except Exception as e:
                print(f"Error generating document summary: {str(e)}")
    
    def retrieve_relevant_context(self, query, k=3):
        """Retrieve relevant context based on query"""
        if not self.documents:
            return []
            
        if self.use_embeddings and self.db:
            try:
                docs = self.db.similarity_search(query, k=k)
                return docs
            except Exception as e:
                print(f"Error in similarity search: {str(e)}")
                print("Falling back to keyword-based search.")
                
        # Keyword-based search as fallback
        results = []
        query_words = set(query.lower().split())
        
        for doc in self.documents:
            score = 0
            text = doc.page_content.lower()
            
            # Count matching words
            for word in query_words:
                if word in text:
                    score += text.count(word)
            
            if score > 0:
                results.append((doc, score))
                
        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in results[:k]]
    
    def get_document_summary(self):
        """Get a summary of processed documents"""
        if "main" in self.summaries:
            return self.summaries["main"]
            
        # Create a basic summary if LLM wasn't available
        summary = f"Documents contain references to {len(self.people_mentioned)} people"
        if self.people_mentioned:
            summary += f" including {', '.join(list(self.people_mentioned)[:5])}"
        if self.dates_mentioned:
            summary += f" with dates like {', '.join(list(self.dates_mentioned)[:3])}"
        if self.locations_mentioned:
            summary += f" in locations such as {', '.join(list(self.locations_mentioned)[:3])}"
            
        return summary

# Session Management
class SessionManager:
    def __init__(self, llm=None):
        """Initialize session manager"""
        self.llm = llm
        self.conversation_history = []
        self.session_summary = ""
        self.session_metadata = {
            "start_time": None,
            "last_update_time": None,
            "session_number": 1,
            "therapy_focus": "general reminiscence",
            "user_mood_trend": [],
            "character_persona": None
        }
        
    def start_session(self, character_persona=None):
        """Start a new therapy session"""
        import datetime
        
        self.conversation_history = []
        self.session_metadata["start_time"] = datetime.datetime.now()
        self.session_metadata["last_update_time"] = datetime.datetime.now()
        self.session_metadata["character_persona"] = character_persona
        
        return {
            "session_id": f"session_{self.session_metadata['session_number']}",
            "start_time": self.session_metadata["start_time"],
            "character_persona": character_persona
        }
    
    def add_message(self, role, content, emotion=None, intensity=None):
        """Add a message to the conversation history"""
        import datetime
        
        # Update last update time
        self.session_metadata["last_update_time"] = datetime.datetime.now()
        
        # Add message to history
        self.conversation_history.append(f"{role}: {content}")
        
        # Track user emotions if provided
        if role.lower() == "patient" and emotion and intensity is not None:
            self.session_metadata["user_mood_trend"].append({
                "emotion": emotion,
                "intensity": intensity,
                "timestamp": datetime.datetime.now()
            })
    
    def get_recent_conversation(self, max_turns=10):
        """Get the recent conversation history"""
        return "\n".join(self.conversation_history[-max_turns:])
    
    def get_full_conversation(self):
        """Get the full conversation history"""
        return "\n".join(self.conversation_history)
    
    def summarize_session(self):
        """Summarize the current session"""
        if not self.llm:
            return "Session summary not available without LLM."
            
        if not self.conversation_history:
            return "No conversation has occurred in this session yet."
            
        try:
            full_conversation = self.get_full_conversation()
            
            prompt = f"""
            A therapist and a patient talked today and had the following conversation:
            
            ====== Conversation Begin ======
            {full_conversation}
            ====== Conversation End ======
            
            {"" if not self.session_summary else f"In their previous session, {self.session_summary}"}
            
            Summarize the interactions between the therapist and the patient. 
            Include key details about main topics, emotions expressed, memories discussed, 
            progress made, and potential areas for future exploration.
            """
            
            summary = self.llm.predict(text=prompt)
            self.session_summary = summary
            return summary
            
        except Exception as e:
            print(f"Error summarizing session: {str(e)}")
            return "Unable to generate session summary."
    
    def end_session(self):
        """End the current session and generate summary"""
        summary = self.summarize_session()
        self.session_metadata["session_number"] += 1
        
        return {
            "session_summary": summary,
            "session_duration": self._calculate_duration(),
            "message_count": len(self.conversation_history),
            "emotion_trend": self._analyze_emotion_trend()
        }
    
    def _calculate_duration(self):
        """Calculate session duration"""
        if (self.session_metadata["start_time"] and 
            self.session_metadata["last_update_time"]):
            duration = self.session_metadata["last_update_time"] - self.session_metadata["start_time"]
            return str(duration)
        return "Unknown"
    
    def _analyze_emotion_trend(self):
        """Analyze emotion trend during session"""
        emotions = self.session_metadata["user_mood_trend"]
        if not emotions:
            return "No emotion data available"
            
        # Get the dominant emotion
        emotion_counts = {}
        for item in emotions:
            emotion = item["emotion"]
            if emotion in emotion_counts:
                emotion_counts[emotion] += 1
            else:
                emotion_counts[emotion] = 1
                
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        
        # Check if intensity changed
        if len(emotions) >= 2:
            start_intensity = emotions[0]["intensity"]
            end_intensity = emotions[-1]["intensity"]
            
            if end_intensity > start_intensity + 0.2:
                intensity_trend = "increased"
            elif end_intensity < start_intensity - 0.2:
                intensity_trend = "decreased"
            else:
                intensity_trend = "remained stable"
                
            return f"Dominant emotion was {dominant_emotion}, intensity {intensity_trend} during the session"
        
        return f"Dominant emotion was {dominant_emotion}"

# Character Persona Adapter
class PersonaAdapter:
    """Adapts therapeutic responses to match a specific character persona"""
    def __init__(self, llm=None):
        self.llm = llm
        self.character_profiles = {
            "iron_man": {
                "name": "Tony Stark (Iron Man)",
                "traits": "confident, witty, intelligent, caring beneath a sarcastic exterior",
                "speaking_style": "uses technical terms, makes pop culture references, occasionally sarcastic but ultimately supportive",
                "catchphrases": ["Sometimes you gotta run before you can walk", "Part of the journey is the end"]
            },
            "captain_america": {
                "name": "Steve Rogers (Captain America)",
                "traits": "honest, principled, compassionate, optimistic",
                "speaking_style": "direct, sincere, occasionally old-fashioned phrases, respectful",
                "catchphrases": ["I can do this all day", "The price of freedom is high"]
            },
            "spider_man": {
                "name": "Peter Parker (Spider-Man)",
                "traits": "friendly, relatable, optimistic, responsible",
                "speaking_style": "casual, conversational, uses humor to connect, youthful enthusiasm",
                "catchphrases": ["With great power comes great responsibility", "I'm just your friendly neighborhood Spider-Man"]
            },
            "wonder_woman": {
                "name": "Diana Prince (Wonder Woman)",
                "traits": "compassionate, powerful, wise, diplomatic",
                "speaking_style": "direct, inspiring, balanced perspective, empowering",
                "catchphrases": ["Fighting doesn't make you a hero", "It's about what you believe"]
            }
        }

    def adapt_response(self, therapeutic_response, character_key, maintain_therapeutic_value=True):
        """Adapt a therapeutic response to match a character's persona"""
        if not self.llm:
            return therapeutic_response
            
        if character_key not in self.character_profiles:
            return therapeutic_response

        character = self.character_profiles[character_key]

        try:
            adaptation_prompt = f"""
            You are an expert dialogue writer for the character {character['name']}.

            Character traits: {character['traits']}
            Speaking style: {character['speaking_style']}
            Catchphrases (use sparingly): {', '.join(character['catchphrases'])}

            Below is a therapeutic message written by a mental health professional:

            "{therapeutic_response}"

            Rewrite this message in the authentic voice of {character['name']}, while maintaining:
            1. The therapeutic value and empathetic qualities
            2. The core message and any questions being asked
            3. The supportive nature of the response

            Your task is to make this sound like {character['name']} is speaking, not to change the substance of the therapy.
            """

            adapted_response = self.llm.predict(text=adaptation_prompt)
            return adapted_response
        except Exception as e:
            print(f"Error adapting response to character persona: {str(e)}")
            return therapeutic_response

# Image Analysis for Multimodal Capability
class ImageAnalyzer:
    def __init__(self, llm=None, use_azure_vision=False, azure_vision_key=None, azure_vision_endpoint=None):
        """Initialize image analyzer"""
        self.llm = llm
        self.use_azure_vision = use_azure_vision
        self.azure_vision_key = azure_vision_key
        self.azure_vision_endpoint = azure_vision_endpoint
        
        # Try to initialize a local vision model if Azure not configured
        if not use_azure_vision:
            try:
                self.vision_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
                print("Local vision model loaded successfully.")
            except Exception as e:
                print(f"Error loading local vision model: {str(e)}")
                self.vision_model = None
    
    def analyze_image(self, image_path):
        """Analyze an image and return a description"""
        if not os.path.exists(image_path):
            return "Image file not found."
            
        # Try Azure Computer Vision if configured
        if self.use_azure_vision:
            try:
                import requests
                # Code to call Azure Computer Vision API
                headers = {
                    'Ocp-Apim-Subscription-Key': self.azure_vision_key,
                    'Content-Type': 'application/octet-stream'
                }
                
                with open(image_path, 'rb') as image_data:
                    response = requests.post(
                        f"{self.azure_vision_endpoint}/vision/v3.1/analyze?visualFeatures=Description,Objects,Faces",
                        headers=headers,
                        data=image_data
                    )
                    
                if response.status_code == 200:
                    result = response.json()
                    description = result.get('description', {}).get('captions', [{}])[0].get('text', "No description available.")
                    return description
                else:
                    print(f"Azure Vision API error: {response.status_code}")
                    return "Error analyzing image with Azure."
                    
            except Exception as e:
                print(f"Error with Azure Vision API: {str(e)}")
                
        # Fall back to local model if available
        if self.vision_model:
            try:
                from PIL import Image
                image = Image.open(image_path)
                result = self.vision_model(image)
                
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict) and 'generated_text' in result[0]:
                        return result[0]['generated_text']
                return "The model was unable to generate a description for this image."
                
            except Exception as e:
                print(f"Error with local vision model: {str(e)}")
                
        # Fall back to LLM with image encoding if all else fails
        if self.llm:
            try:
                # Encode image to base64
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Create a prompt asking LLM to imagine what's in the image
                prompt = f"""
                I have an image that I'm looking at. Based on the filename '{os.path.basename(image_path)}',
                please suggest what this image might show. Think about what would be most relevant in the
                context of reminiscence therapy. What memories might this image evoke if it shows:
                
                1. A family gathering
                2. A special place or landmark
                3. An important life event
                4. A cherished possession
                
                Please note that you can't actually see the image, so your description is speculative,
                but should offer thoughtful possibilities that could help in a therapeutic conversation.
                """
                
                response = self.llm.predict(text=prompt)
                return f"[Image analysis]: {response}"
                
            except Exception as e:
                print(f"Error with image encoding fallback: {str(e)}")
                
        return "Unable to analyze image with available tools."
    
    def incorporate_image_into_therapy(self, image_description, user_query):
        """Generate therapeutic questions based on image description"""
        if not self.llm:
            return [
                "What feelings does this image bring up for you?",
                "Does this image remind you of any specific memories?",
                "How does this image connect to your life journey?"
            ]
            
        try:
            prompt = f"""
            In a reminiscence therapy session, the patient has shared an image.
            
            Image description: {image_description}
            
            Patient's comment about the image: {user_query}
            
            Generate 3 therapeutic questions that could help the patient explore memories
            and emotions connected to this image. These questions should:
            1. Be open-ended and encourage storytelling
            2. Connect to emotional experiences
            3. Help explore how past experiences shape current identity
            
            Format your response as a numbered list of questions only.
            """
            
            response = self.llm.predict(text=prompt)
            
            # Extract questions from response
            questions = []
            for line in response.strip().split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    question_text = re.sub(r'^\d+\.\s*|-\s*', '', line).strip()
                    if question_text:
                        questions.append(question_text)
            
            # Ensure we have at least one question
            if not questions:
                questions = ["What feelings does this image bring up for you?"]
                
            return questions
            
        except Exception as e:
            print(f"Error generating image-based questions: {str(e)}")
            return [
                "What feelings does this image bring up for you?",
                "Does this image remind you of any specific memories?",
                "How does this image connect to your life journey?"
            ]

# Agent System for Character Research
class CharacterResearchAgent:
    def __init__(self, llm=None):
        """Initialize character research agent"""
        self.llm = llm
        self.character_database = {
            "iron_man": {
                "real_name": "Tony Stark",
                "traits": "genius inventor, billionaire, philanthropist, reformed arms dealer",
                "background": "Born to wealthy industrialist Howard Stark, Tony inherited Stark Industries and transformed it from a weapons manufacturer to a technology innovator. After being captured and wounded, he built the Iron Man suit to escape and decided to use his technology to protect others.",
                "personality": "witty, sarcastic, confident, sometimes arrogant, deeply caring beneath a tough exterior",
                "catchphrases": ["I am Iron Man", "Sometimes you gotta run before you can walk", "Part of the journey is the end"],
                "therapeutic_approach": "Would use humor to disarm, practical advice, and technological metaphors. Might suggest building or creating as therapeutic activities."
            },
            "captain_america": {
                "real_name": "Steve Rogers",
                "traits": "super-soldier, leader, symbol of hope, man out of time",
                "background": "Born during the Great Depression, Steve was a frail young man determined to serve his country in WWII. Selected for the super-soldier program, he became Captain America but was frozen for decades before being revived in the modern world.",
                "personality": "principled, honest, determined, sometimes old-fashioned, unfailingly loyal",
                "catchphrases": ["I can do this all day", "The price of freedom is high", "I'm with you 'til the end of the line"],
                "therapeutic_approach": "Would emphasize values, resilience, and finding meaning in hardship. Would listen attentively and offer sincere encouragement focused on inner strength."
            },
            "spider_man": {
                "real_name": "Peter Parker",
                "traits": "friendly neighborhood hero, scientist, photographer",
                "background": "Orphaned at a young age and raised by his Aunt May and Uncle Ben, Peter was bitten by a radioactive spider and gained spider-like abilities. After his uncle was killed, he learned that 'with great power comes great responsibility' and became Spider-Man.",
                "personality": "friendly, witty, relatable, sometimes insecure, deeply compassionate",
                "catchphrases": ["With great power comes great responsibility", "Just your friendly neighborhood Spider-Man"],
                "therapeutic_approach": "Would use relatability and humor to connect, normalize struggles, and emphasize the importance of community and asking for help."
            }
        }
        
    def research_character(self, character_name):
        """Research a character for role-playing therapy"""
        # Check if character is in database
        character_key = character_name.lower().replace(" ", "_")
        if character_key in self.character_database:
            return self.character_database[character_key]
            
        # If not in database and LLM available, try to generate information
        if self.llm:
            try:
                prompt = f"""
                Create a character profile for {character_name} that could be used in role-playing therapy:
                
                1. Provide the character's real name
                2. List key character traits (3-5 traits)
                3. Write a brief background (2-3 sentences)
                4. Describe their personality (4-5 descriptive terms)
                5. Include 2-3 catchphrases
                6. Suggest how they might approach therapy (2-3 sentences)
                
                Format your response as JSON with the following keys:
                "real_name", "traits", "background", "personality", "catchphrases" (as an array), and "therapeutic_approach"
                """
                
                response = self.llm.predict(text=prompt)
                
                # Try to parse JSON from response
                try:
                    import json
                    # Find JSON in the response
                    json_match = re.search(r'({[\s\S]*})', response)
                    if json_match:
                        json_str = json_match.group(1)
                        character_data = json.loads(json_str)
                        return character_data
                except:
                    print(f"Failed to parse JSON for character {character_name}")
                    
            except Exception as e:
                print(f"Error researching character {character_name}: {str(e)}")
                
        # Return a generic profile if all else fails
        return {
            "real_name": character_name,
            "traits": "supportive, understanding, compassionate",
            "background": "A character known for their ability to connect with others and provide guidance.",
            "personality": "friendly, approachable, empathetic, insightful",
            "catchphrases": ["I'm here to help", "Let's explore that together"],
            "therapeutic_approach": "Uses a warm, supportive approach focused on active listening and empathetic responses."
        }
        
    def get_character_therapy_style(self, character_name):
        """Get therapy style suggestions for a character"""
        character_data = self.research_character(character_name)
        
        if not self.llm:
            return character_data.get("therapeutic_approach", "No specific therapeutic approach information available.")
            
        try:
            prompt = f"""
            Based on this character profile:
            
            Name: {character_data.get('real_name', character_name)}
            Traits: {character_data.get('traits', 'N/A')}
            Background: {character_data.get('background', 'N/A')}
            Personality: {character_data.get('personality', 'N/A')}
            Catchphrases: {', '.join(character_data.get('catchphrases', ['N/A']))}
            
            Provide guidance on how this character would approach therapy:
            
            1. What therapeutic techniques would align with their personality?
            2. How would their communication style manifest in a therapy context?
            3. What unique perspective or approach might they bring to therapy?
            4. What types of metaphors or examples would they use?
            
            Focus specifically on how they would conduct reminiscence therapy, which involves
            helping people recall and explore their past experiences in a positive way.
            """
            
            response = self.llm.predict(text=prompt)
            return response
            
        except Exception as e:
            print(f"Error getting therapy style for {character_name}: {str(e)}")
            return character_data.get("therapeutic_approach", "No specific therapeutic approach information available.")

# Main Reminiscence Therapy System
class ReminiscenceTherapySystem:
    def __init__(self, api_key=None, api_base=None, api_version=None, deployment_name=None):
        """Initialize the complete reminiscence therapy system"""
        # Initialize LLM provider (if API key provided)
        if api_key:
            self.llm_provider = AzureLLMProvider(
                api_key=api_key,
                api_base=api_base,
                api_version=api_version,
                deployment_name=deployment_name
            )
            self.llm = self.llm_provider.llm
        else:
            self.llm_provider = None
            self.llm = None
            print("Running in minimal API mode. Some features will be limited.")
            
        # Initialize all components
        self.memory_graph = MemoryGraph(use_api=(self.llm is not None), llm=self.llm)
        self.empathetic_engagement = EmpatheticEngagement(llm=self.llm)
        self.document_processor = DocumentProcessor(use_embeddings=True, llm=self.llm)
        self.session_manager = SessionManager(llm=self.llm)
        self.persona_adapter = PersonaAdapter(llm=self.llm)
        self.image_analyzer = ImageAnalyzer(llm=self.llm)
        self.character_agent = CharacterResearchAgent(llm=self.llm)
        
        # System settings
        self.character_persona = None
        self.use_character_persona = False
        self.system_initialized = True
        
    def process_document(self, file_path):
        """Process an uploaded document for RAG"""
        return self.document_processor.process_document(file_path)
    
    def process_image(self, image_path):
        """Process an uploaded image"""
        return self.image_analyzer.analyze_image(image_path)
    
    def set_character_persona(self, character_name):
        """Set a character persona for role-playing therapy"""
        self.character_persona = character_name
        self.use_character_persona = True
        
        # Research the character
        character_info = self.character_agent.research_character(character_name)
        
        return {
            "character": character_name,
            "real_name": character_info.get("real_name", character_name),
            "traits": character_info.get("traits", ""),
            "therapeutic_approach": character_info.get("therapeutic_approach", "")
        }
    
    def clear_character_persona(self):
        """Clear the character persona"""
        self.character_persona = None
        self.use_character_persona = False
    
    def start_session(self):
        """Start a new therapy session"""
        return self.session_manager.start_session(character_persona=self.character_persona)
    
    def end_session(self):
        """End the current therapy session"""
        return self.session_manager.end_session()
    
    def visualize_memory_graph(self, output_file="memory_graph.png"):
        """Visualize the memory graph"""
        graph_buffer = self.memory_graph.visualize_graph(output_file=output_file)
        return graph_buffer
    
    def generate_response(self, user_input, is_image=False):
        """Generate a response to user input"""
        # If input is from an image, process differently
        if is_image:
            image_description = self.process_image(user_input)
            questions = self.image_analyzer.incorporate_image_into_therapy(
                image_description, 
                "Tell me about this image"
            )
            
            # Use the first question as response, save others for follow-up
            response = f"I see this image. {image_description}\n\n{questions[0]}"
            
            # Add to conversation history
            self.session_manager.add_message("Patient", f"[Shared an image]")
            self.session_manager.add_message("Therapist", response)
            
            return {
                "response": response,
                "image_description": image_description,
                "follow_up_questions": questions[1:] if len(questions) > 1 else []
            }
            
        # Process text input
        emotion, intensity = self.empathetic_engagement.detect_emotion(user_input)
        
        # Add user input to conversation history
        self.session_manager.add_message("Patient", user_input, emotion, intensity)
        
        # Get recent conversation
        recent_conversation = self.session_manager.get_recent_conversation()
        
        # Extract memory nodes (if any conversation has occurred)
        if len(self.session_manager.conversation_history) > 1:
            full_conversation = self.session_manager.get_full_conversation()
            events = self.memory_graph.extract_events_from_conversation(full_conversation)
            self.memory_graph.add_events_to_graph(events)
        
        # Get relevant memories from the graph
        relevant_memories = self.memory_graph.get_relevant_memories(user_input)
        
        # Get relevant document contexts
        relevant_docs = self.document_processor.retrieve_relevant_context(user_input)
        doc_context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Create session context
        session_context = {
            "session_number": self.session_manager.session_metadata["session_number"],
            "therapy_focus": self.session_manager.session_metadata["therapy_focus"],
            "document_summary": self.document_processor.get_document_summary() if relevant_docs else ""
        }
        
        # Generate basic therapeutic response
        base_response = self.empathetic_engagement.generate_response(
            user_input=user_input,
            conversation_history=recent_conversation,
            memory_nodes=relevant_memories,
            character_persona=self.character_persona if self.use_character_persona else None,
            session_context=session_context
        )
        
        # Adapt response to character persona if enabled
        if self.use_character_persona and self.character_persona:
            response = self.persona_adapter.adapt_response(
                base_response,
                self.character_persona.lower().replace(" ", "_")
            )
        else:
            response = base_response
            
        # Add to conversation history
        self.session_manager.add_message("Therapist", response)
        
        # Generate follow-up questions based on memory graph
        follow_up_questions = self.memory_graph.generate_follow_up_questions(max_questions=3)
        
        return {
            "response": response,
            "emotion": emotion,
            "intensity": intensity,
            "follow_up_questions": follow_up_questions
        }
    
    

# Example usage
def start_conversation(therapy_system):
    """Start a conversation with the therapy system"""
    print("Therapist: Hello! I'm here to talk with you today. How are you feeling?")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Therapist: It was nice talking with you. Take care!")
            summary = therapy_system.end_session()
            print("\nSession Summary:")
            print(summary["session_summary"])
            break

        result = therapy_system.generate_response(user_input)
        print(f"Therapist: {result['response']}")

# Command Line Interface
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Reminiscence Therapy System")
    parser.add_argument("--api_key", help="Azure OpenAI API Key")
    parser.add_argument("--api_base", help="Azure OpenAI API Base URL")
    parser.add_argument("--api_version", help="Azure OpenAI API Version")
    parser.add_argument("--deployment", help="Azure OpenAI Deployment Name")
    parser.add_argument("--character", help="Character persona (e.g., 'Iron Man')")
    parser.add_argument("--document", help="Path to document for processing")
    parser.add_argument("--minimal", action="store_true", help="Run in minimal API mode")
    
    args = parser.parse_args()
    
    # Initialize the therapy system
    if args.minimal:
        # Minimal API mode (no Azure)
        therapy_system = ReminiscenceTherapySystem()
    else:
        # Full mode with Azure
        therapy_system = ReminiscenceTherapySystem(
            api_key=args.api_key or os.environ.get("AZURE_OPENAI_API_KEY"),
            api_base=args.api_base or os.environ.get("AZURE_OPENAI_API_BASE"),
            api_version=args.api_version or os.environ.get("AZURE_OPENAI_API_VERSION"),
            deployment_name=args.deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT_NAME")
        )
    
    # Set character persona if provided
    if args.character:
        therapy_system.set_character_persona(args.character)
        print(f"Character persona set to: {args.character}")
    
    # Process document if provided
    if args.document:
        therapy_system.process_document(args.document)
        print(f"Processed document: {args.document}")
    
    # Start therapy session
    therapy_system.start_session()
    start_conversation(therapy_system)

if __name__ == "__main__":
    main()