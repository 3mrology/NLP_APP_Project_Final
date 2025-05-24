import fix_pwd
import os
import json
import re
import random
from typing import Dict, List, Any, Optional, Tuple

class CharacterPersona:
    """
    Represents a character persona with traits, speaking style, and therapeutic approach
    """
    def __init__(
        self,
        name: str,
        real_name: str = "",
        traits: str = "",
        background: str = "",
        personality: str = "",
        catchphrases: List[str] = None,
        therapeutic_approach: str = "",
        profile_image: str = None
    ):
        self.name = name
        self.real_name = real_name or name
        self.traits = traits
        self.background = background
        self.personality = personality
        self.catchphrases = catchphrases or []
        self.therapeutic_approach = therapeutic_approach
        self.profile_image = profile_image
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert persona to dictionary"""
        return {
            "name": self.name,
            "real_name": self.real_name,
            "traits": self.traits,
            "background": self.background,
            "personality": self.personality,
            "catchphrases": self.catchphrases,
            "therapeutic_approach": self.therapeutic_approach,
            "profile_image": self.profile_image
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CharacterPersona':
        """Create persona from dictionary"""
        return cls(
            name=data.get("name", ""),
            real_name=data.get("real_name", ""),
            traits=data.get("traits", ""),
            background=data.get("background", ""),
            personality=data.get("personality", ""),
            catchphrases=data.get("catchphrases", []),
            therapeutic_approach=data.get("therapeutic_approach", ""),
            profile_image=data.get("profile_image")
        )


class RolePlayingEngine:
    """
    Engine to manage character personas and adapt therapeutic responses
    """
    def __init__(self, llm=None):
        self.llm = llm
        self.active_persona = None
        self.personas = self._load_default_personas()
        self.response_styles = {
            "supportive": self._generate_supportive_style,
            "analytical": self._generate_analytical_style,
            "motivational": self._generate_motivational_style,
            "reflective": self._generate_reflective_style,
            "humorous": self._generate_humorous_style
        }
        
    def _load_default_personas(self) -> Dict[str, CharacterPersona]:
        """Load default character personas"""
        default_personas = {
            "iron_man": CharacterPersona(
                name="Iron Man",
                real_name="Tony Stark",
                traits="genius inventor, billionaire, philanthropist, reformed arms dealer",
                background="Born to wealthy industrialist Howard Stark, Tony inherited Stark Industries and transformed it from a weapons manufacturer to a technology innovator. After being captured and wounded, he built the Iron Man suit to escape and decided to use his technology to protect others.",
                personality="witty, sarcastic, confident, sometimes arrogant, deeply caring beneath a tough exterior",
                catchphrases=["I am Iron Man", "Sometimes you gotta run before you can walk", "Part of the journey is the end"],
                therapeutic_approach="Uses humor to disarm, practical advice, and technological metaphors. Might suggest building or creating as therapeutic activities.",
                profile_image="iron_man.jpg"
            ),
            "captain_america": CharacterPersona(
                name="Captain America",
                real_name="Steve Rogers",
                traits="super-soldier, leader, symbol of hope, man out of time",
                background="Born during the Great Depression, Steve was a frail young man determined to serve his country in WWII. Selected for the super-soldier program, he became Captain America but was frozen for decades before being revived in the modern world.",
                personality="principled, honest, determined, sometimes old-fashioned, unfailingly loyal",
                catchphrases=["I can do this all day", "The price of freedom is high", "I'm with you 'til the end of the line"],
                therapeutic_approach="Emphasizes values, resilience, and finding meaning in hardship. Listens attentively and offers sincere encouragement focused on inner strength.",
                profile_image="captain_america.jpg"
            ),
            "spider_man": CharacterPersona(
                name="Spider-Man",
                real_name="Peter Parker",
                traits="friendly neighborhood hero, scientist, photographer",
                background="Orphaned at a young age and raised by his Aunt May and Uncle Ben, Peter was bitten by a radioactive spider and gained spider-like abilities. After his uncle was killed, he learned that 'with great power comes great responsibility' and became Spider-Man.",
                personality="friendly, witty, relatable, sometimes insecure, deeply compassionate",
                catchphrases=["With great power comes great responsibility", "Just your friendly neighborhood Spider-Man"],
                therapeutic_approach="Uses relatability and humor to connect, normalizes struggles, and emphasizes the importance of community and asking for help.",
                profile_image="spider_man.jpg"
            ),
            "wonder_woman": CharacterPersona(
                name="Wonder Woman",
                real_name="Diana Prince",
                traits="Amazon warrior, diplomat, symbol of truth and justice",
                background="Born on the hidden island of Themyscira as an Amazon princess, Diana left her home to help mankind during a great war. She brings the values of her people – truth, compassion, and justice – to a complex modern world.",
                personality="compassionate, wise, determined, curious about humanity, direct yet empathetic",
                catchphrases=["We fight when there is no other choice", "It's not about what you deserve, it's about what you believe", "I will fight for those who cannot fight for themselves"],
                therapeutic_approach="Combines direct truth-telling with deep compassion. Encourages facing difficult truths while emphasizing hope and inner strength.",
                profile_image="wonder_woman.jpg"
            ),
            "batman": CharacterPersona(
                name="Batman",
                real_name="Bruce Wayne",
                traits="detective, strategist, vigilante, businessman",
                background="After witnessing his parents' murder as a child, Bruce dedicated his life to fighting crime in Gotham City. He trained his mind and body to peak condition and uses his wealth to create advanced technology to fight as Batman.",
                personality="analytical, brooding, determined, strategic, deeply empathetic despite his stoic exterior",
                catchphrases=["I'm Batman", "It's not who I am underneath, but what I do that defines me", "Why do we fall? So we can learn to pick ourselves up"],
                therapeutic_approach="Uses methodical problem-solving and analytical thinking. Helps identify root causes of issues and develop step-by-step strategies for overcoming challenges.",
                profile_image="batman.jpg"
            )
        }
        return default_personas
    
    def add_persona(self, persona: CharacterPersona) -> None:
        """Add a new character persona"""
        key = persona.name.lower().replace(" ", "_")
        self.personas[key] = persona
        
    def get_persona(self, name: str) -> Optional[CharacterPersona]:
        """Get a character persona by name"""
        key = name.lower().replace(" ", "_")
        return self.personas.get(key)
    
    def list_personas(self) -> List[str]:
        """List all available personas"""
        return sorted(self.personas.keys())
    
    def set_active_persona(self, name: str) -> bool:
        """Set the active character persona"""
        key = name.lower().replace(" ", "_")
        if key in self.personas:
            self.active_persona = key
            return True
        return False
    
    def clear_active_persona(self) -> None:
        """Clear the active character persona"""
        self.active_persona = None
    
    def get_active_persona(self) -> Optional[CharacterPersona]:
        """Get the currently active persona"""
        if self.active_persona:
            return self.personas.get(self.active_persona)
        return None
    
    def save_personas(self, file_path: str = "character_personas.json") -> None:
        """Save personas to a JSON file"""
        try:
            data = {key: persona.to_dict() for key, persona in self.personas.items()}
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved {len(data)} personas to {file_path}")
        except Exception as e:
            print(f"Error saving personas: {str(e)}")
    
    def load_personas(self, file_path: str = "character_personas.json") -> bool:
        """Load personas from a JSON file"""
        if not os.path.exists(file_path):
            return False
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for key, persona_data in data.items():
                self.personas[key] = CharacterPersona.from_dict(persona_data)
                
            print(f"Loaded {len(data)} personas from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading personas: {str(e)}")
            return False
    
    def research_character(self, name: str) -> Optional[CharacterPersona]:
        """Research a character using the LLM"""
        if not self.llm:
            print("LLM not available for character research")
            return None
            
        try:
            prompt = f"""
            Create a character profile for {name} that could be used in role-playing therapy:
            
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
                # Find JSON in the response
                json_match = re.search(r'({[\s\S]*})', response)
                if json_match:
                    json_str = json_match.group(1)
                    data = json.loads(json_str)
                    
                    persona = CharacterPersona(
                        name=name,
                        real_name=data.get("real_name", name),
                        traits=data.get("traits", ""),
                        background=data.get("background", ""),
                        personality=data.get("personality", ""),
                        catchphrases=data.get("catchphrases", []),
                        therapeutic_approach=data.get("therapeutic_approach", "")
                    )
                    
                    # Add to personas
                    key = name.lower().replace(" ", "_")
                    self.personas[key] = persona
                    return persona
            except Exception as e:
                print(f"Failed to parse JSON for character {name}: {str(e)}")
                
        except Exception as e:
            print(f"Error researching character {name}: {str(e)}")
            
        return None
    
    def adapt_response(
        self, 
        response: str, 
        persona_name: Optional[str] = None, 
        style: str = "supportive", 
        maintain_therapeutic_value: bool = True
    ) -> str:
        """Adapt a therapeutic response to match a character's persona"""
        if not self.llm:
            return response
            
        # Determine which persona to use
        persona_key = None
        if persona_name:
            persona_key = persona_name.lower().replace(" ", "_")
        elif self.active_persona:
            persona_key = self.active_persona
            
        if not persona_key or persona_key not in self.personas:
            return response
            
        persona = self.personas[persona_key]
        
        try:
            # Generate style-specific prompt
            style_generator = self.response_styles.get(style, self.response_styles["supportive"])
            prompt = style_generator(response, persona)
            
            # Call LLM for adaptation
            adapted_response = self.llm.predict(text=prompt)
            return adapted_response
            
        except Exception as e:
            print(f"Error adapting response: {str(e)}")
            return response
    
    def _generate_supportive_style(self, response: str, persona: CharacterPersona) -> str:
        """Generate prompt for supportive style"""
        return f"""
        You are an expert dialogue writer for the character {persona.name} ({persona.real_name}).

        Character traits: {persona.traits}
        Personality: {persona.personality}
        Catchphrases (use sparingly): {', '.join(persona.catchphrases)}

        Below is a therapeutic message written by a mental health professional:

        "{response}"

        Rewrite this message in the authentic voice of {persona.name}, while maintaining:
        1. The therapeutic value and empathetic qualities
        2. The core message and any questions being asked
        3. The supportive nature of the response

        Your task is to make this sound like {persona.name} is speaking, not to change the substance of the therapy.
        Make sure to keep the same therapeutic intent while adapting the language, tone, and expression to match the character.
        """
    
    def _generate_analytical_style(self, response: str, persona: CharacterPersona) -> str:
        """Generate prompt for analytical style"""
        return f"""
        You are an expert dialogue writer for the character {persona.name} ({persona.real_name}).

        Character traits: {persona.traits}
        Personality: {persona.personality}
        Catchphrases (use sparingly): {', '.join(persona.catchphrases)}

        Below is a therapeutic message written by a mental health professional:

        "{response}"

        Rewrite this message in the authentic voice of {persona.name}, emphasizing a more analytical approach:
        1. Break down complex emotional concepts into logical components
        2. Add more structure and clarity to the response
        3. Use the character's specific way of analyzing situations
        4. Maintain the therapeutic core while making it sound like the character's analytical style

        Make it sound like {persona.name} is approaching this therapeutically but through their unique analytical lens.
        """
    
    def _generate_motivational_style(self, response: str, persona: CharacterPersona) -> str:
        """Generate prompt for motivational style"""
        return f"""
        You are an expert dialogue writer for the character {persona.name} ({persona.real_name}).

        Character traits: {persona.traits}
        Personality: {persona.personality}
        Catchphrases (use sparingly): {', '.join(persona.catchphrases)}

        Below is a therapeutic message written by a mental health professional:

        "{response}"

        Rewrite this message in the authentic voice of {persona.name}, adding a motivational flavor:
        1. Emphasize strengths, possibilities, and potential
        2. Add encouragement and inspiration while maintaining authenticity
        3. Use the character's unique motivational style and language
        4. Keep the therapeutic core while making it more energizing and activating

        Make it sound like {persona.name} is genuinely trying to motivate and inspire the person through their unique perspective.
        """
    
    def _generate_reflective_style(self, response: str, persona: CharacterPersona) -> str:
        """Generate prompt for reflective style"""
        return f"""
        You are an expert dialogue writer for the character {persona.name} ({persona.real_name}).

        Character traits: {persona.traits}
        Personality: {persona.personality}
        Catchphrases (use sparingly): {', '.join(persona.catchphrases)}

        Below is a therapeutic message written by a mental health professional:

        "{response}"

        Rewrite this message in the authentic voice of {persona.name}, emphasizing a more reflective approach:
        1. Include more thoughtful pauses and considered perspectives
        2. Draw on the character's personal experiences or background when relevant
        3. Make connections between different ideas in a contemplative way
        4. Maintain the therapeutic value while adding more depth and introspection

        Make it sound like {persona.name} is speaking thoughtfully and reflectively, drawing from their experiences and wisdom.
        """
    
    def _generate_humorous_style(self, response: str, persona: CharacterPersona) -> str:
        """Generate prompt for humorous style"""
        return f"""
        You are an expert dialogue writer for the character {persona.name} ({persona.real_name}).

        Character traits: {persona.traits}
        Personality: {persona.personality}
        Catchphrases (use sparingly): {', '.join(persona.catchphrases)}

        Below is a therapeutic message written by a mental health professional:

        "{response}"

        Rewrite this message in the authentic voice of {persona.name}, adding appropriate humor:
        1. Include the character's style of humor (wit, sarcasm, puns, etc.)
        2. Add light-hearted elements that would be typical of this character
        3. Use humor to make the therapeutic message more engaging and relatable
        4. Ensure the humor is appropriate and doesn't undermine the therapeutic value

        Make it sound like {persona.name} is being genuine and supportive, but with their characteristic sense of humor.
        """
    
    def generate_character_introduction(self, persona_name: Optional[str] = None) -> str:
        """Generate a character introduction for the start of therapy"""
        # Determine which persona to use
        persona_key = None
        if persona_name:
            persona_key = persona_name.lower().replace(" ", "_")
        elif self.active_persona:
            persona_key = self.active_persona
            
        if not persona_key or persona_key not in self.personas:
            return "Hello! I'm here to talk with you today. How are you feeling?"
            
        persona = self.personas[persona_key]
        
        if not self.llm:
            # Default introduction if LLM not available
            catchphrase = random.choice(persona.catchphrases) if persona.catchphrases else ""
            return f"Hey there! I'm {persona.name}. {catchphrase} I'm here to talk with you today. How are you feeling?"
        
        try:
            prompt = f"""
            You're writing an introduction for a therapy session where the therapist is taking on the persona of {persona.name} ({persona.real_name}).

            Character traits: {persona.traits}
            Personality: {persona.personality}
            Catchphrases (use sparingly): {', '.join(persona.catchphrases)}
            Background: {persona.background}

            Write a brief, friendly introduction (2-4 sentences) where {persona.name} introduces themselves to the patient
            and invites them to start the therapy session. Make it sound authentic to the character while being
            appropriately therapeutic and welcoming.
            """
            
            introduction = self.llm.predict(text=prompt)
            return introduction
            
        except Exception as e:
            print(f"Error generating character introduction: {str(e)}")
            
            # Fallback introduction
            catchphrase = random.choice(persona.catchphrases) if persona.catchphrases else ""
            return f"Hey there! I'm {persona.name}. {catchphrase} I'm here to talk with you today. How are you feeling?"


class TherapeuticDialogueTemplate:
    """
    Stores and manages dialogue templates for therapy sessions
    """
    def __init__(self):
        self.templates = self._load_default_templates()
        
    def _load_default_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load default dialogue templates"""
        return {
            "greetings": {
                "default": [
                    "Hello! How are you feeling today?",
                    "Welcome to our session. How has your day been so far?",
                    "It's good to see you. How are you doing today?",
                    "Thank you for joining me today. How are you feeling?"
                ],
                "iron_man": [
                    "Hey there! Tony Stark reporting for duty. How's life treating you?",
                    "Welcome to Stark Therapy. No robots or suits today, just us talking. How've you been?",
                    "Let's dive right in. What's going on in that head of yours today?"
                ],
                "captain_america": [
                    "Good to see you. How are things on your front today?",
                    "I believe in taking time to check in with ourselves. How are you doing today?",
                    "Sometimes the strongest thing we can do is talk about how we're feeling. So, how are you?"
                ]
            },
            "reflection": {
                "default": [
                    "It sounds like you're feeling {emotion} about {topic}.",
                    "I'm hearing that {topic} has been making you feel {emotion}.",
                    "So from what you're sharing, {topic} brings up feelings of {emotion} for you."
                ],
                "iron_man": [
                    "So if I'm running the diagnostics right, {topic} has your emotional circuits running {emotion}.",
                    "Let me get this straight - {topic} has you feeling {emotion}? I've been there.",
                    "JARVIS doesn't need to analyze this one - clearly {topic} has you feeling {emotion}."
                ],
                "captain_america": [
                    "Back in my day, we didn't always talk about feelings, but that was a mistake. I can see {topic} has you feeling {emotion}.",
                    "I understand battlefields, and it sounds like {topic} has become one where you're feeling {emotion}.",
                    "A good soldier recognizes their feelings. {topic} has you feeling {emotion}, and that's important to acknowledge."
                ]
            },
            "questions": {
                "default": [
                    "Can you tell me more about {topic}?",
                    "How did you feel when {topic} happened?",
                    "What thoughts came up for you during {topic}?",
                    "How has {topic} affected other areas of your life?"
                ],
                "iron_man": [
                    "Let's reverse-engineer this {topic} situation. What exactly went down?",
                    "If we were to analyze {topic} like one of my suit prototypes, what would the blueprints show?",
                    "I'm curious - and I'm rarely without answers - what made {topic} hit you that way?"
                ],
                "captain_america": [
                    "I've found that understanding our history helps us move forward. Can you tell me more about {topic}?",
                    "Even the strongest shield can't block everything. How did {topic} get past your defenses?",
                    "Sometimes the right strategy requires understanding the full situation. What else should I know about {topic}?"
                ]
            },
            "encouragement": {
                "default": [
                    "You're showing real strength by discussing this.",
                    "I appreciate your openness. That takes courage.",
                    "The work you're doing here is important.",
                    "You've shown remarkable resilience through this."
                ],
                "iron_man": [
                    "Not everyone has what it takes to face this stuff. You do. Impressed.",
                    "Take it from someone who's been in tough spots - you're handling this better than you think.",
                    "If mental fortitude could be manufactured, yours would be worth billions. Just saying."
                ],
                "captain_america": [
                    "What you're doing takes real courage - the kind that matters most.",
                    "I've seen soldiers in battle with less bravery than you're showing right now.",
                    "Keep pushing forward. That's how we win the battles that really matter."
                ]
            },
            "closing": {
                "default": [
                    "Thank you for sharing today. Let's continue next time.",
                    "We've covered a lot today. I look forward to our next session.",
                    "This was a meaningful session. I appreciate your openness.",
                    "I think we've made good progress today. Let's build on this next time."
                ],
                "iron_man": [
                    "Well, that's our time. Even billionaires have schedules. You did good work today.",
                    "Jarvis is telling me we need to wrap up. But this? This was good. Let's pick it up next time.",
                    "The mark of a good session isn't how you feel now, it's the upgrades you make later. See you next time."
                ],
                "captain_america": [
                    "I think we've accomplished our mission for today. Rest up, and we'll continue next time.",
                    "You've fought well today. Sometimes the hardest battles are the ones we fight with ourselves.",
                    "In my time, we've learned that progress takes patience. You're on the right path. Until next time."
                ]
            }
        }
    
    def get_template(self, category: str, persona: str = "default", replacements: Dict[str, str] = None) -> str:
        """Get a random template from a category, optionally for a specific persona"""
        if category not in self.templates:
            return ""
            
        if persona not in self.templates[category]:
            persona = "default"
            
        templates = self.templates[category][persona]
        if not templates:
            return ""
            
        template = random.choice(templates)
        
        # Replace placeholders if replacements provided
        if replacements:
            for key, value in replacements.items():
                template = template.replace("{" + key + "}", value)
                
        return template
    
    def add_template(self, category: str, persona: str, template: str) -> None:
        """Add a new template"""
        if category not in self.templates:
            self.templates[category] = {}
            
        if persona not in self.templates[category]:
            self.templates[category][persona] = []
            
        self.templates[category][persona].append(template)
    
    def save_templates(self, file_path: str = "dialogue_templates.json") -> None:
        """Save templates to a JSON file"""
        try:
            with open(file_path, 'w') as f:
                json.dump(self.templates, f, indent=2)
            print(f"Saved templates to {file_path}")
        except Exception as e:
            print(f"Error saving templates: {str(e)}")
    
    def load_templates(self, file_path: str = "dialogue_templates.json") -> bool:
        """Load templates from a JSON file"""
        if not os.path.exists(file_path):
            return False
            
        try:
            with open(file_path, 'r') as f:
                self.templates = json.load(f)
            print(f"Loaded templates from {file_path}")
            return True
        except Exception as e:
            print(f"Error loading templates: {str(e)}")
            return False


def main():
    """Test function for role-playing module"""
    # Initialize role-playing engine
    engine = RolePlayingEngine()
    
    # Print available personas
    print("Available personas:")
    for key in engine.list_personas():
        persona = engine.get_persona(key)
        print(f"- {persona.name} ({persona.real_name})")
    
    # Test dialogue templates
    print("\nTesting dialogue templates:")
    templates = TherapeuticDialogueTemplate()
    
    print("\nDefault greeting:")
    print(templates.get_template("greetings", "default"))
    
    print("\nIron Man greeting:")
    print(templates.get_template("greetings", "iron_man"))
    
    print("\nIron Man reflection with replacements:")
    print(templates.get_template("reflection", "iron_man", {"topic": "losing your job", "emotion": "anxious"}))
    
    print("\nCaptain America encouragement:")
    print(templates.get_template("encouragement", "captain_america"))


if __name__ == "__main__":
    main()