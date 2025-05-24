import os
import re
import base64
import json
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import numpy as np

# For document loading and processing
try:
    from langchain.document_loaders import TextLoader, PyPDFLoader, CSVLoader, Docx2txtLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: LangChain not available. Document processing will be limited.")

# For image processing
try:
    from PIL import Image
    from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: Transformers not available. Image processing will be limited.")

# NLP utilities
try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    # Download required NLTK resources
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Text analysis will be limited.")


class MemoryExtractor:
    """
    Extracts potential memories, people, places, and dates from text
    """
    def __init__(self):
        # Initialize regex patterns
        self.date_patterns = [
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,\s+\d{4}\b',  # January 1st, 2020
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',  # 01/01/2020
            r'\bin (\d{4})\b',  # in 2020
            r'\b(19|20)\d{2}\b',  # 1990 or 2020
            r'\b(yesterday|today|tomorrow|last week|last month|last year)\b',  # Relative dates
            r'\b(spring|summer|fall|winter|autumn) (?:of )?(\d{4})?\b',  # Seasons with optional year
            r'\b(childhood|youth|teenage years|college|university|school days)\b'  # Life periods
        ]
        
        self.person_patterns = [
            r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b',  # First Last
            r'\bmy (mother|father|brother|sister|aunt|uncle|cousin|grandmother|grandfather|grandma|grandpa|wife|husband|partner|boyfriend|girlfriend|son|daughter|child|friend|colleague|boss|teacher|neighbor)\b',  # Relationships
            r'\b(mom|dad|sis|bro|granny|grandad|grammy|grampa)\b'  # Informal relationships
        ]
        
        self.location_patterns = [
            r'\bin ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b',  # In LocationName
            r'\bto ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b',  # To LocationName
            r'\bfrom ([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b',  # From LocationName
            r'\bat (the|my|our) ([a-z]+)\b',  # At the/my/our location
            r'\b(house|home|apartment|school|college|university|hospital|park|beach|mountain|lake|river|ocean|sea|city|town|village|country|continent|planet)\b'  # Common locations
        ]
        
        self.event_patterns = [
            r'\b(birthday|wedding|anniversary|graduation|funeral|holiday|vacation|trip|journey|visit|move|promotion|retirement|birth|death|marriage|divorce|engagement|proposal)\b',  # Life events
            r'\b(celebrated|attended|went to|visited|traveled|moved|started|finished|graduated|married|divorced|engaged|proposed|born|died)\b'  # Event verbs
        ]
        
        # Emotion keywords for memory significance
        self.emotion_words = [
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
        
        # Life stages for chronological ordering
        self.life_stages = [
            "birth", "infancy", "childhood", "elementary school", "middle school", 
            "high school", "college", "university", "twenties", "thirties", 
            "forties", "fifties", "sixties", "seventies", "eighties", "nineties"
        ]
    
    def extract_dates(self, text: str) -> List[str]:
        """Extract dates from text"""
        dates = []
        for pattern in self.date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                dates.append(match.group(0))
        return dates
    
    def extract_people(self, text: str) -> List[str]:
        """Extract people mentions from text"""
        people = []
        for pattern in self.person_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # For relationship patterns, capture the whole phrase
                if pattern.startswith(r'\bmy '):
                    full_match = match.group(0)
                    people.append(full_match)
                else:
                    # For proper names, just get the name
                    if match.group(1)[0].isupper():
                        people.append(match.group(1))
        return people
    
    def extract_locations(self, text: str) -> List[str]:
        """Extract location mentions from text"""
        locations = []
        for pattern in self.location_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Depending on the pattern, extract the right group
                if pattern.startswith(r'\bat '):
                    # For "at the/my/our location" pattern
                    full_match = match.group(0)
                    locations.append(full_match)
                elif pattern.startswith(r'\b(house|'):
                    # For common location terms
                    locations.append(match.group(1))
                else:
                    # For proper location names, get the name
                    try:
                        location = match.group(1)
                        if location[0].isupper():
                            locations.append(location)
                    except IndexError:
                        pass
        return locations
    
    def extract_events(self, text: str) -> List[str]:
        """Extract event mentions from text"""
        events = []
        for pattern in self.event_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Get the sentence containing the event for context
                sentences = sent_tokenize(text) if NLTK_AVAILABLE else [text]
                for sentence in sentences:
                    if match.group(0) in sentence:
                        events.append(sentence)
                        break
        return events
    
    def extract_potential_memories(self, text: str) -> List[Dict[str, Any]]:
        """Extract potential memories from text"""
        # Split text into sentences if NLTK is available
        if NLTK_AVAILABLE:
            sentences = sent_tokenize(text)
        else:
            # Fallback to simple sentence splitting
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
        
        memories = []
        
        for sentence in sentences:
            # Check for memory indicators
            dates = self.extract_dates(sentence)
            people = self.extract_people(sentence)
            locations = self.extract_locations(sentence)
            
            # Check for emotion words
            has_emotion = any(word in sentence.lower() for word in self.emotion_words)
            
            # Define memory significance
            memory_significance = 0
            memory_significance += len(dates) * 2
            memory_significance += len(people) * 1.5
            memory_significance += len(locations) * 1
            memory_significance += 2 if has_emotion else 0
            
            # Only consider as potential memory if it has some significance
            if memory_significance > 0:
                # Determine topic
                topic = self.determine_topic(sentence)
                
                # Determine timeframe
                timeframe = self.determine_timeframe(sentence, dates)
                
                memories.append({
                    "text": sentence,
                    "dates": dates,
                    "people": people,
                    "locations": locations,
                    "topic": topic,
                    "timeframe": timeframe,
                    "significance": memory_significance
                })
        
        # Sort by significance
        memories.sort(key=lambda x: x["significance"], reverse=True)
        return memories
    
    def determine_topic(self, text: str) -> str:
        """Determine the most likely topic for the text"""
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
            return "General Memory"
            
        max_category = max(category_scores, key=category_scores.get)
        return max_category.capitalize()
    
    def determine_timeframe(self, text: str, dates: List[str]) -> str:
        """Determine the timeframe of a memory"""
        # If explicit dates are mentioned, use the first one
        if dates:
            return dates[0]
            
        # Check for life stage indicators
        text_lower = text.lower()
        for stage in self.life_stages:
            if stage in text_lower:
                return f"During {stage}"
                
        # Check for relative time indicators
        relative_time_patterns = [
            (r'\b(few|couple|several|many) (days|weeks|months|years) ago\b', 'Recent past'),
            (r'\blast (week|month|year)\b', 'Recent past'),
            (r'\bwhen I was (young|a child|a kid|a teenager|in school|in college)\b', 'Youth'),
            (r'\bgrowing up\b', 'Childhood'),
            (r'\b(recently|lately|these days)\b', 'Present'),
            (r'\bin the (past|old days)\b', 'Past')
        ]
        
        for pattern, timeframe in relative_time_patterns:
            if re.search(pattern, text_lower):
                return timeframe
                
        # Default to unknown timeframe
        return "Unknown timeframe"


class DocumentProcessor:
    """
    Processes documents for RAG-enhanced therapy
    """
    def __init__(self, use_embeddings: bool = False, llm=None):
        self.use_embeddings = use_embeddings and LANGCHAIN_AVAILABLE
        self.llm = llm
        self.memory_extractor = MemoryExtractor()
        
        # Initialize components if available
        if LANGCHAIN_AVAILABLE:
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            if use_embeddings:
                try:
                    self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                    print("Embeddings model loaded successfully.")
                except Exception as e:
                    print(f"Error loading embeddings model: {str(e)}")
                    self.use_embeddings = False
            
            self.vector_store = None
        
        # Document storage
        self.documents = []
        self.chunks = []
        self.document_metadata = {}
        self.extracted_info = {
            "people": set(),
            "dates": set(),
            "locations": set(),
            "topics": set(),
            "events": set(),
            "memories": []
        }
    
    def process_file(self, file_path: str) -> bool:
        """Process a document file"""
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False
        
        try:
            # Determine file type and use appropriate loader
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if LANGCHAIN_AVAILABLE:
                # Use LangChain loaders if available
                if file_extension == ".pdf":
                    loader = PyPDFLoader(file_path)
                elif file_extension == ".csv":
                    loader = CSVLoader(file_path)
                elif file_extension in [".docx", ".doc"]:
                    loader = Docx2txtLoader(file_path)
                else:
                    # Default to text loader
                    loader = TextLoader(file_path)
                
                documents = loader.load()
                
                # Split into chunks
                chunks = self.text_splitter.split_documents(documents)
                
                # Store documents and chunks
                self.documents.extend(documents)
                self.chunks.extend(chunks)
                
                # Create vector store if using embeddings
                if self.use_embeddings:
                    if self.vector_store is None:
                        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
                    else:
                        self.vector_store.add_documents(chunks)
                
                # Extract information from the documents
                self._extract_information(documents)
                
                # Store metadata
                doc_id = len(self.document_metadata)
                self.document_metadata[doc_id] = {
                    "file_path": file_path,
                    "file_name": os.path.basename(file_path),
                    "file_type": file_extension,
                    "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "num_chunks": len(chunks),
                    "size_bytes": os.path.getsize(file_path)
                }
                
                print(f"Processed document: {file_path}")
                return True
                
            else:
                # Fallback for when LangChain is not available
                print("Processing document without LangChain...")
                
                # Simple text extraction based on file type
                text = ""
                try:
                    if file_extension.lower() in [".txt", ".md", ".csv"]:
                        with open(file_path, 'r', encoding='utf-8') as file:
                            text = file.read()
                    elif file_extension.lower() == ".pdf":
                        print("PDF processing requires LangChain. Basic metadata extraction only.")
                        text = f"PDF file: {os.path.basename(file_path)}"
                    else:
                        print(f"Unsupported file type without LangChain: {file_extension}")
                        text = f"Unsupported file: {os.path.basename(file_path)}"
                
                    # Extract basic information without LangChain
                    self.documents.append({"page_content": text, "metadata": {"source": file_path}})
                    self._extract_information_fallback(text, file_path)
                    
                    # Store basic metadata
                    doc_id = len(self.document_metadata)
                    self.document_metadata[doc_id] = {
                        "file_path": file_path,
                        "file_name": os.path.basename(file_path),
                        "file_type": file_extension,
                        "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "size_bytes": os.path.getsize(file_path)
                    }
                    
                    print(f"Basic processing of document: {file_path}")
                    return True
                    
                except Exception as e:
                    print(f"Error in fallback document processing: {str(e)}")
                    return False
        
        except Exception as e:
            print(f"Error processing document {file_path}: {str(e)}")
            return False
    
    def _extract_information(self, documents: List[Dict[str, Any]]) -> None:
        """Extract meaningful information from documents using LangChain"""
        combined_text = ""
        for doc in documents:
            combined_text += doc.page_content + "\n\n"
        
        # Extract entities
        memories = self.memory_extractor.extract_potential_memories(combined_text)
        
        # Add to extracted info
        for memory in memories:
            self.extracted_info["people"].update(memory["people"])
            self.extracted_info["dates"].update(memory["dates"])
            self.extracted_info["locations"].update(memory["locations"])
            self.extracted_info["topics"].add(memory["topic"])
            
            # Store the memory
            self.extracted_info["memories"].append(memory)
        
        # Create a summary using LLM if available
        if self.llm and len(combined_text) > 0:
            try:
                prompt = f"""
                Please analyze this document content and extract key information for reminiscence therapy:

                {combined_text[:4000]}  # Limit to first 4000 chars to avoid context limits

                1. Summarize in 2-3 sentences what this document contains
                2. List any key people mentioned 
                3. List any important places mentioned
                4. List any significant dates or time periods referenced
                5. Identify any potential significant memories described

                Format your response in JSON with these keys: 
                "summary", "people", "places", "dates", "potential_memories"
                """
                
                response = self.llm.predict(text=prompt)
                
                # Try to parse JSON from response
                try:
                    # Find JSON in the response
                    json_match = re.search(r'({[\s\S]*})', response)
                    if json_match:
                        json_str = json_match.group(1)
                        data = json.loads(json_str)
                        
                        # Store additional extracted information
                        if "people" in data and isinstance(data["people"], list):
                            self.extracted_info["people"].update(data["people"])
                        if "places" in data and isinstance(data["places"], list):
                            self.extracted_info["locations"].update(data["places"])
                        if "dates" in data and isinstance(data["dates"], list):
                            self.extracted_info["dates"].update(data["dates"])
                        if "potential_memories" in data and isinstance(data["potential_memories"], list):
                            self.extracted_info["events"].update(data["potential_memories"])
                        
                        # Store the summary
                        if "summary" in data:
                            self.extracted_info["summary"] = data["summary"]
                            
                except Exception as e:
                    print(f"Failed to parse LLM extraction response: {str(e)}")
            
            except Exception as e:
                print(f"Error using LLM for information extraction: {str(e)}")
    
    def _extract_information_fallback(self, text: str, file_path: str) -> None:
        """Basic information extraction without LangChain"""
        # Extract memories using the memory extractor
        memories = self.memory_extractor.extract_potential_memories(text)
        
        # Add to extracted info
        for memory in memories:
            self.extracted_info["people"].update(memory["people"])
            self.extracted_info["dates"].update(memory["dates"])
            self.extracted_info["locations"].update(memory["locations"])
            self.extracted_info["topics"].add(memory["topic"])
            
            # Store the memory
            self.extracted_info["memories"].append(memory)
    
    def retrieve_relevant_context(self, query: str, k: int = 3) -> List[Dict[str, str]]:
        """Retrieve relevant context based on query"""
        if not self.documents:
            return []
        
        if self.use_embeddings and self.vector_store:
            try:
                docs = self.vector_store.similarity_search(query, k=k)
                return [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
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
                results.append(({"content": doc.page_content, "metadata": doc.metadata}, score))
        
        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in results[:k]]
    
    def get_document_summary(self) -> Dict[str, Any]:
        """Get a summary of processed documents"""
        summary = {
            "num_documents": len(self.document_metadata),
            "document_names": [metadata["file_name"] for metadata in self.document_metadata.values()],
            "total_chunks": len(self.chunks),
            "people_mentioned": list(self.extracted_info["people"])[:10],  # Limit to top 10
            "locations_mentioned": list(self.extracted_info["locations"])[:10],
            "dates_mentioned": list(self.extracted_info["dates"])[:10],
            "topics": list(self.extracted_info["topics"]),
            "num_potential_memories": len(self.extracted_info["memories"]),
            "summary": self.extracted_info.get("summary", "No summary available")
        }
        return summary
    
    def get_potential_memories(self, max_memories: int = 10) -> List[Dict[str, Any]]:
        """Get potential memories extracted from documents"""
        # Sort by significance
        memories = sorted(self.extracted_info["memories"], key=lambda x: x["significance"], reverse=True)
        return memories[:max_memories]
    
    def get_memory_related_to_query(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Find memories related to a specific query"""
        if not self.extracted_info["memories"]:
            return []
            
        # If LLM is available, use semantic search
        if self.llm:
            try:
                # Format memories for evaluation
                memories_text = ""
                for i, memory in enumerate(self.extracted_info["memories"]):
                    memories_text += f"{i+1}. {memory['text']}\n"
                
                prompt = f"""
                Given this query: "{query}"
                
                Find the most relevant memories from this list:
                {memories_text}
                
                Return only the numbers of the top {max_results} most relevant memories, 
                separated by commas. For example: "2, 5, 7"
                """
                
                response = self.llm.predict(text=prompt)
                
                # Extract indices
                indices = []
                for match in re.finditer(r'\b\d+\b', response):
                    index = int(match.group()) - 1
                    if 0 <= index < len(self.extracted_info["memories"]):
                        indices.append(index)
                
                # Get the memories
                return [self.extracted_info["memories"][i] for i in indices[:max_results]]
                
            except Exception as e:
                print(f"Error finding memories with LLM: {str(e)}")
        
        # Fallback: Simple keyword matching
        relevant_memories = []
        query_words = set(query.lower().split())
        
        for memory in self.extracted_info["memories"]:
            score = 0
            memory_text = memory["text"].lower()
            
            # Count matching words
            for word in query_words:
                if word in memory_text:
                    score += memory_text.count(word)
            
            # Check topics
            if "topic" in memory and memory["topic"].lower() in query.lower():
                score += 3
                
            # Check people
            for person in memory.get("people", []):
                if person.lower() in query.lower():
                    score += 2
            
            # Check dates
            for date in memory.get("dates", []):
                if date.lower() in query.lower():
                    score += 2
                    
            # Check locations
            for location in memory.get("locations", []):
                if location.lower() in query.lower():
                    score += 2
            
            if score > 0:
                relevant_memories.append((memory, score))
        
        # Sort by relevance
        relevant_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in relevant_memories[:max_results]]


class ImageProcessor:
    """
    Processes images for multimodal therapy
    """
    def __init__(self, llm=None, use_vision_model: bool = True):
        self.llm = llm
        self.use_vision_model = use_vision_model and TRANSFORMERS_AVAILABLE
        
        # Initialize vision model if available
        if self.use_vision_model:
            try:
                self.vision_model = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
                print("Vision model loaded successfully.")
            except Exception as e:
                print(f"Error loading vision model: {str(e)}")
                self.use_vision_model = False
        
        # Image storage
        self.processed_images = {}
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Process an image and extract information"""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return {
                "image_path": image_path,
                "status": "error",
                "description": "Image file not found"
            }
            
        try:
            image_id = len(self.processed_images)
            
            # Basic image metadata
            image_metadata = {
                "image_id": image_id,
                "image_path": image_path,
                "file_name": os.path.basename(image_path),
                "processed_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "size_bytes": os.path.getsize(image_path)
            }
            
            description = "No description available"
            
            # Extract description with vision model if available
            if self.use_vision_model:
                try:
                    image = Image.open(image_path)
                    
                    # Get image dimensions
                    width, height = image.size
                    image_metadata["dimensions"] = f"{width}x{height}"
                    
                    # Generate caption
                    result = self.vision_model(image)
                    
                    if isinstance(result, list) and len(result) > 0:
                        if isinstance(result[0], dict) and 'generated_text' in result[0]:
                            description = result[0]['generated_text']
                        else:
                            description = str(result[0])
                            
                except Exception as e:
                    print(f"Error with vision model: {str(e)}")
                    
            # Fall back to LLM with image name if vision model fails
            if description == "No description available" and self.llm:
                try:
                    prompt = f"""
                    I have an image with the filename '{os.path.basename(image_path)}'.
                    Based on the filename, suggest what this image might show.
                    Provide a brief description (1-2 sentences) that might be helpful
                    in a reminiscence therapy context.
                    """
                    
                    description = self.llm.predict(text=prompt)
                    description = f"[Based on filename] {description}"
                    
                except Exception as e:
                    print(f"Error with LLM fallback: {str(e)}")
            
            # Store result
            result = {
                **image_metadata,
                "description": description,
                "status": "success"
            }
            
            self.processed_images[image_id] = result
            return result
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return {
                "image_path": image_path,
                "status": "error",
                "description": f"Error: {str(e)}"
            }
    
    def generate_therapeutic_questions(self, image_description: str, user_query: str = "", max_questions: int = 3) -> List[str]:
        """Generate therapeutic questions based on image description"""
        if not self.llm:
            # Default questions if LLM not available
            return [
                "What feelings does this image bring up for you?",
                "Does this image remind you of any specific memories?",
                "How does this image connect to your life journey?"
            ]
            
        try:
            user_context = f"The patient commented: '{user_query}'" if user_query else ""
            
            prompt = f"""
            In a reminiscence therapy session, the patient has shared an image.
            
            Image description: {image_description}
            
            {user_context}
            
            Generate {max_questions} therapeutic questions that could help the patient explore memories
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
                    if question_text and question_text[-1] not in '.?!':
                        question_text += '?'
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
    
    def get_image_as_base64(self, image_path: str) -> Optional[str]:
        """Convert image to base64 for embedding in HTML/UI"""
        if not os.path.exists(image_path):
            return None
            
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            print(f"Error encoding image: {str(e)}")
            return None
    
    def resize_image(self, image_path: str, max_width: int = 800, max_height: int = 600) -> Optional[str]:
        """Resize image for display"""
        if not os.path.exists(image_path):
            return None
            
        try:
            image = Image.open(image_path)
            
            # Calculate new dimensions
            width, height = image.size
            ratio = min(max_width / width, max_height / height)
            
            # Only resize if image is larger than max dimensions
            if ratio < 1:
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
            # Save to buffer
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            
            return buffer
            
        except Exception as e:
            print(f"Error resizing image: {str(e)}")
            return None


def main():
    """Test function for document and image processing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Document and Image Processing")
    parser.add_argument("--document", help="Path to document for processing")
    parser.add_argument("--image", help="Path to image for processing")
    
    args = parser.parse_args()
    
    # Test document processing
    if args.document:
        print(f"Processing document: {args.document}")
        doc_processor = DocumentProcessor(use_embeddings=False)
        success = doc_processor.process_file(args.document)
        
        if success:
            summary = doc_processor.get_document_summary()
            print("\nDocument Summary:")
            for key, value in summary.items():
                if isinstance(value, list):
                    print(f"{key}: {', '.join(value[:5])}{'...' if len(value) > 5 else ''}")
                else:
                    print(f"{key}: {value}")
                    
            print("\nPotential Memories:")
            memories = doc_processor.get_potential_memories(5)
            for i, memory in enumerate(memories):
                print(f"{i+1}. {memory['text']}")
                print(f"   Topic: {memory['topic']}, Timeframe: {memory['timeframe']}")
                print(f"   People: {', '.join(memory['people']) if memory['people'] else 'None'}")
                print()
    
    # Test image processing
    if args.image:
        print(f"Processing image: {args.image}")
        img_processor = ImageProcessor(use_vision_model=True)
        result = img_processor.process_image(args.image)
        
        print("\nImage Processing Result:")
        for key, value in result.items():
            print(f"{key}: {value}")
            
        print("\nSuggested Questions:")
        questions = img_processor.generate_therapeutic_questions(result["description"])
        for i, question in enumerate(questions):
            print(f"{i+1}. {question}")


if __name__ == "__main__":
    main()