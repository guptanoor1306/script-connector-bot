#!/usr/bin/env python3
"""
Script Connector Bot for Zero1 by Zerodha
Analyzes video scripts to identify missing connectors and validate existing ones.
Connectors should link sections back to intro or payoff to retain viewers.
"""

import re
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI

class ConnectorType(Enum):
    INTRO = "intro"
    PAYOFF = "payoff"
    SECTION_TO_SECTION = "section_to_section"
    MISSING = "missing"

@dataclass
class ScriptSection:
    """Represents a section in the script"""
    number: int
    title: str
    content: str
    start_line: int
    end_line: int

@dataclass
class Connector:
    """Represents a connector in the script"""
    text: str
    type: ConnectorType
    section_before: Optional[int]
    section_after: Optional[int]
    line_number: int
    is_valid: bool
    issues: List[str]

@dataclass
class ScriptAnalysis:
    """Complete analysis of a script"""
    intro: str
    payoff: str
    sections: List[ScriptSection]
    connectors: List[Connector]
    missing_connectors: List[Tuple[int, int]]  # (section_before, section_after)
    suggestions: List[str]
    score: float  # 0-100, higher is better

class ScriptConnectorBot:
    """Main bot class for analyzing script connectors"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.intro_keywords = [
            'intro', 'introduction', 'hook', 'opening', 'start', 'beginning'
        ]
        self.payoff_keywords = [
            'payoff', 'conclusion', 'ending', 'final', 'wrap up', 'closing', 'end'
        ]
        self.connector_indicators = [
            'connecting line', 'connector', 'transition', 'bridge', 'link',
            'now', 'but', 'however', 'meanwhile', 'furthermore', 'moreover',
            'that brings us to', 'this leads us to', 'speaking of', 'on that note'
        ]
        
        # Initialize OpenAI client if API key is provided
        self.openai_client = None
        if openai_api_key or os.getenv('OPENAI_API_KEY'):
            try:
                self.openai_client = OpenAI(api_key=openai_api_key or os.getenv('OPENAI_API_KEY'))
            except TypeError as e:
                # Fallback for compatibility issues
                print(f"OpenAI client initialization warning: {e}")
                self.openai_client = None
        
    def parse_script(self, script_text: str, custom_intro: str = None) -> ScriptAnalysis:
        """Parse the script and extract all components step by step"""
        lines = script_text.split('\n')
        
        print("=== STEP 1: EXTRACTING INTRO ===")
        if custom_intro:
            intro = custom_intro
            print(f"Using custom intro: {len(intro)} characters")
            print(f"Custom intro: {intro}")
        else:
            intro = self._extract_intro(script_text)
            print(f"Intro found: {len(intro)} characters")
        if intro:
            print(f"Intro preview: {intro[:200]}...")
        else:
            print("No intro found - this may affect connector suggestions")
        
        print("\n=== STEP 2: EXTRACTING PAYOFF ===")
        payoff = self._extract_payoff(script_text)
        print(f"Payoff found: {len(payoff)} characters")
        if payoff:
            print(f"Payoff preview: {payoff[:200]}...")
        
        print("\n=== STEP 3: EXTRACTING SECTIONS ===")
        sections = self._extract_sections(script_text, lines)
        print(f"Sections found: {len(sections)}")
        for section in sections:
            print(f"  Section {section.number}: {section.title}")
        
        print("\n=== STEP 4: EXTRACTING CONNECTORS ===")
        connectors = self._extract_connectors(script_text, lines, sections)
        print(f"Connectors found: {len(connectors)}")
        for connector in connectors:
            print(f"  Connector: {connector.text[:100]}...")
        
        print("\n=== STEP 5: FINDING MISSING CONNECTORS ===")
        missing_connectors = self._find_missing_connectors(sections, connectors)
        print(f"Missing connectors: {len(missing_connectors)}")
        for before, after in missing_connectors:
            print(f"  Need connector between Section {before} and Section {after}")
        
        print("\n=== STEP 6: GENERATING SUGGESTIONS ===")
        suggestions = self._generate_suggestions(sections, connectors, missing_connectors, intro, payoff, script_text)
        print(f"Suggestions generated: {len(suggestions)}")
        
        print("\n=== STEP 7: CALCULATING SCORE ===")
        score = self._calculate_score(sections, connectors, missing_connectors)
        print(f"Final score: {score:.1f}/100")
        
        return ScriptAnalysis(
            intro=intro,
            payoff=payoff,
            sections=sections,
            connectors=connectors,
            missing_connectors=missing_connectors,
            suggestions=suggestions,
            score=score
        )
    
    def _extract_intro(self, script_text: str) -> str:
        """Extract the intro section with better pattern matching"""
        lines = script_text.split('\n')
        
        # Look for intro patterns more comprehensively
        intro_patterns = [
            r'(?i)(intro[:\-]?\s*[^\n]*?)(?=\n\n|\nSection|\nChapter|\n\d+\.|$)',
            r'(?i)(introduction[:\-]?\s*[^\n]*?)(?=\n\n|\nSection|\nChapter|\n\d+\.|$)',
            r'(?i)(hook[:\-]?\s*[^\n]*?)(?=\n\n|\nSection|\nChapter|\n\d+\.|$)',
        ]
        
        # First, try to find explicit intro markers
        for pattern in intro_patterns:
            matches = re.finditer(pattern, script_text, re.MULTILINE | re.DOTALL)
            for match in matches:
                intro_text = match.group(1).strip()
                if len(intro_text) > 30:  # Ensure it's substantial
                    return intro_text
        
        # Fallback: take first substantial lines that seem like intro
        intro_lines = []
        for i, line in enumerate(lines[:25]):  # Check more lines
            line_clean = line.strip()
            if line_clean and not line_clean.lower().startswith(('section', 'chapter', '1.', '2.', '3.', 'connecting')):
                # Skip very short lines and lines that look like headers
                if len(line_clean) > 10 and not re.match(r'^[A-Z\s]+$', line_clean):
                    intro_lines.append(line_clean)
                    if len(intro_lines) >= 8:  # Allow longer intro
                        break
        
        return '\n'.join(intro_lines)
    
    def _extract_payoff(self, script_text: str) -> str:
        """Extract the payoff/conclusion section with better pattern matching"""
        lines = script_text.split('\n')
        
        # Look for payoff patterns more comprehensively
        payoff_patterns = [
            r'(?i)(payoff[:\-]?\s*[^\n]*?)(?=\n\n|$)',
            r'(?i)(conclusion[:\-]?\s*[^\n]*?)(?=\n\n|$)',
            r'(?i)(ending[:\-]?\s*[^\n]*?)(?=\n\n|$)',
            r'(?i)(final[:\-]?\s*[^\n]*?)(?=\n\n|$)',
            r'(?i)(wrap[:\-]?\s*up[^\n]*?)(?=\n\n|$)',
        ]
        
        # First, try to find explicit payoff markers
        for pattern in payoff_patterns:
            matches = re.finditer(pattern, script_text, re.MULTILINE | re.DOTALL)
            for match in matches:
                payoff_text = match.group(1).strip()
                if len(payoff_text) > 30:  # Ensure it's substantial
                    return payoff_text
        
        # Fallback: take last substantial lines that seem like payoff
        payoff_lines = []
        for i in range(len(lines) - 1, max(0, len(lines) - 15), -1):
            line_clean = lines[i].strip()
            if line_clean and not line_clean.lower().startswith(('section', 'chapter', 'connecting')):
                payoff_lines.insert(0, line_clean)
                if len(payoff_lines) >= 5:  # Limit payoff length
                    break
        
        return '\n'.join(payoff_lines)
    
    def _extract_sections(self, script_text: str, lines: List[str]) -> List[ScriptSection]:
        """Extract all sections from the script"""
        sections = []
        
        # Handle both normal and spaced-out text formats
        section_patterns = [
            r'Section (\d+)[^\\n]*-([^\\n]*)',  # Normal format
            r'S\s*e\s*c\s*t\s*i\s*o\s*n\s*(\d+)[^\\n]*-([^\\n]*)',  # Spaced format
            r'Section\s*(\d+)[^\\n]*-([^\\n]*)',  # Alternative format
            r'S\s*e\s*c\s*t\s*i\s*o\s*n\s*(\d+)\s*[^\w]*([^\\n]*)'  # More flexible spaced format
        ]
        
        matches = []
        for pattern in section_patterns:
            pattern_matches = list(re.finditer(pattern, script_text, re.IGNORECASE))
            matches.extend(pattern_matches)
        
        # Sort matches by position and remove duplicates
        matches.sort(key=lambda x: x.start())
        
        # Remove duplicate sections (same number and similar position)
        unique_matches = []
        seen_sections = set()
        
        for match in matches:
            section_num = int(match.group(1))
            title = match.group(2).strip()
            # Create a key to identify duplicates
            key = (section_num, title[:20])  # Use first 20 chars of title
            if key not in seen_sections:
                seen_sections.add(key)
                unique_matches.append(match)
        
        for i, match in enumerate(unique_matches):
            section_num = int(match.group(1))
            title = match.group(2).strip()
            start_line = script_text[:match.start()].count('\n')
            
            # Find end line (next section or end of script)
            if i + 1 < len(unique_matches):
                end_line = script_text[:unique_matches[i + 1].start()].count('\n')
            else:
                end_line = len(lines)
            
            # Extract content
            content = script_text[match.start():match.end() + (end_line - start_line) * 50]  # Approximate
            
            sections.append(ScriptSection(
                number=section_num,
                title=title,
                content=content,
                start_line=start_line,
                end_line=end_line
            ))
        
        return sections
    
    def _extract_connectors(self, script_text: str, lines: List[str], sections: List[ScriptSection]) -> List[Connector]:
        """Extract all connectors from the script"""
        connectors = []
        
        print("Looking for connector patterns...")
        
        # Look for explicit connector indicators - handle both normal and spaced formats
        connector_patterns = [
            r'(?i)(connecting line[:\-]?\s*[^\n]*?)(?=\n|Section|$)',  # Normal format with colon/dash
            r'(?i)(connecting line[^\n]*?)(?=\n|$)',  # Normal format
            r'(?i)(c\s*o\s*n\s*n\s*e\s*c\s*t\s*i\s*n\s*g\s*l\s*i\s*n\s*e[^\n]*?)(?=\n|$)',  # Spaced format
            r'(?i)(transition[^\n]*?)(?=\n|$)',  # Alternative
            r'(?i)(bridge[^\n]*?)(?=\n|$)'  # Alternative
        ]
        
        matches = []
        for i, pattern in enumerate(connector_patterns):
            pattern_matches = list(re.finditer(pattern, script_text))
            print(f"Pattern {i+1} found {len(pattern_matches)} matches")
            matches.extend(pattern_matches)
        
        # Also look for implicit connectors (sentences that connect sections)
        print("Looking for implicit connectors...")
        implicit_connectors = self._find_implicit_connectors(script_text, sections)
        print(f"Found {len(implicit_connectors)} implicit connectors")
        
        # Process all found connectors
        for match in matches:
            connector_text = match.group(1).strip()
            if len(connector_text) > 10:  # Ensure it's substantial
                line_number = script_text[:match.start()].count('\n') + 1
                
                # Determine which sections this connector is between
                section_before, section_after = self._find_connector_sections(
                    line_number, sections
                )
                
                # Determine connector type and validation
                connector_type, is_valid, issues = self._analyze_connector(
                    connector_text, sections, line_number, section_before, section_after
                )
                
                connectors.append(Connector(
                    text=connector_text,
                    type=connector_type,
                    section_before=section_before,
                    section_after=section_after,
                    line_number=line_number,
                    is_valid=is_valid,
                    issues=issues
                ))
        
        # Sort matches by position
        matches.sort(key=lambda x: x.start())
        
        for match in matches:
            connector_text = match.group(1).strip()
            line_number = script_text[:match.start()].count('\n')
            
            print(f"Processing connector: {connector_text[:50]}...")
            
            # Determine connector type and validity
            connector_type, is_valid, issues = self._analyze_connector(connector_text, sections, line_number)
            
            # Find which sections this connector bridges
            section_before, section_after = self._find_connector_sections(line_number, sections)
            
            connectors.append(Connector(
                text=connector_text,
                type=connector_type,
                section_before=section_before,
                section_after=section_after,
                line_number=line_number,
                is_valid=is_valid,
                issues=issues
            ))
        
        # Add implicit connectors
        connectors.extend(implicit_connectors)
        
        # Remove duplicate connectors (same line number and similar text)
        unique_connectors = []
        seen_connectors = set()
        
        for connector in connectors:
            # Create a key to identify duplicates
            key = (connector.line_number, connector.text[:50])
            if key not in seen_connectors:
                seen_connectors.add(key)
                unique_connectors.append(connector)
        
        return unique_connectors
    
    def _find_connector_sections(self, line_number: int, sections: List[ScriptSection]) -> Tuple[Optional[int], Optional[int]]:
        """Find which sections a connector is between based on line number"""
        section_before = None
        section_after = None
        
        for i, section in enumerate(sections):
            if section.start_line <= line_number <= section.end_line:
                # Connector is within this section
                if i > 0:
                    section_before = sections[i-1].number
                section_after = section.number
                break
            elif line_number < section.start_line:
                # Connector is before this section
                if i > 0:
                    section_before = sections[i-1].number
                section_after = section.number
                break
        
        return section_before, section_after
    
    def _find_implicit_connectors(self, script_text: str, sections: List[ScriptSection]) -> List[Connector]:
        """Find implicit connectors that don't have explicit labels"""
        connectors = []
        
        # Look for transition sentences between sections
        transition_words = [
            'but', 'however', 'meanwhile', 'furthermore', 'moreover', 'now', 'that brings us to',
            'this leads us to', 'speaking of', 'on that note', 'moving on', 'next', 'then',
            'as we saw', 'remember', 'recall', 'think about', 'imagine', 'consider'
        ]
        
        # Split text into sentences
        sentences = re.split(r'[.!?]+', script_text)
        
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
                
            # Check if sentence contains transition words
            has_transition = any(word in sentence.lower() for word in transition_words)
            
            if has_transition:
                # Check if this sentence is between sections
                sentence_pos = script_text.find(sentence)
                section_before, section_after = self._find_connector_sections(sentence_pos, sections)
                
                if section_before and section_after and section_before != section_after:
                    connector_type, is_valid, issues = self._analyze_connector(sentence, sections, sentence_pos)
                    
                    connectors.append(Connector(
                        text=sentence,
                        type=connector_type,
                        section_before=section_before,
                        section_after=section_after,
                        line_number=sentence_pos,
                        is_valid=is_valid,
                        issues=issues
                    ))
        
        return connectors
    
    def _generate_ai_connector_suggestions(self, script_text: str, sections: List[ScriptSection], intro: str, payoff: str) -> List[str]:
        """Generate AI-powered connector suggestions when no connectors are found"""
        if not self.openai_client:
            return []
        
        try:
            # Prepare context for AI
            sections_text = "\n".join([f"Section {s.number}: {s.title}" for s in sections])
            
            prompt = f"""
            Analyze this video script and suggest compelling connectors between sections. 
            The script is about road accidents and needs connectors that retain viewer engagement.
            
            Script sections:
            {sections_text}
            
            Intro: {intro[:200] if intro else "No intro found"}
            Payoff: {payoff[:200] if payoff else "No payoff found"}
            
            Please suggest 3-5 specific connector sentences that:
            1. Link sections back to the intro story
            2. Build anticipation for the payoff
            3. Create curiosity and suspense
            4. Use direct audience engagement
            5. Include transition words like "But here's the thing...", "However...", "What's interesting is..."
            
            Format each suggestion as: "Connector between Section X and Y: [sentence]"
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7
            )
            
            suggestions = response.choices[0].message.content.strip().split('\n')
            return [s.strip() for s in suggestions if s.strip()]
            
        except Exception as e:
            print(f"AI suggestion generation failed: {e}")
            return []
    
    def _segment_text_intelligently(self, text: str) -> List[str]:
        """Segment text into logical, complete sentences using AI"""
        if not self.openai_client:
            # Fallback to simple sentence splitting
            import re
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
        
        try:
            prompt = f"""
            Segment this text into logical, complete sentences. 
            Each segment should be a complete thought that makes sense on its own.
            Return each complete sentence on a new line.
            
            Text to segment:
            {text[:800]}  # Limit to avoid token limits
            
            Format: One complete sentence per line.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.3
            )
            
            segments = response.choices[0].message.content.strip().split('\n')
            return [s.strip() for s in segments if s.strip()]
            
        except Exception as e:
            print(f"AI text segmentation failed: {e}")
            # Fallback to simple sentence splitting
            import re
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _generate_specific_connector_suggestions(self, sections: List[ScriptSection], intro: str, payoff: str, script_text: str) -> List[str]:
        """Generate specific connector text suggestions for missing connectors"""
        suggestions = []
        
        # Generate specific connector suggestions for each section transition
        for i in range(len(sections) - 1):
            current_section = sections[i]
            next_section = sections[i + 1]
            
            print(f"Generating suggestions for Section {current_section.number} → {next_section.number}")
            print(f"Current section: {current_section.title}")
            print(f"Next section: {next_section.title}")
            print(f"Intro available: {len(intro) if intro else 0} characters")
            
            # Create specific connector suggestions based on section content
            connector_suggestions = self._create_connector_for_sections(
                current_section, next_section, intro, payoff, script_text
            )
            suggestions.extend(connector_suggestions)
        
        return suggestions
    
    def _create_connector_for_sections(self, section_before: ScriptSection, section_after: ScriptSection, intro: str, payoff: str, script_text: str) -> List[str]:
        """Create specific connector text for a section transition using AI analysis"""
        suggestions = []
        
        # Use AI to analyze the content and generate contextual connectors
        if self.openai_client:
            try:
                # Extract relevant content around the sections
                before_content = self._extract_section_content(script_text, section_before)
                after_content = self._extract_section_content(script_text, section_after)
                
                prompt = f"""
                Analyze these two script sections and create 3 compelling connector sentences that would bridge them effectively.
                
                Section {section_before.number} ({section_before.title}):
                {before_content}
                
                Section {section_after.number} ({section_after.title}):
                {after_content}
                
                Intro context: {intro[:300] if intro else "No intro provided"}
                Payoff context: {payoff[:300] if payoff else "No payoff provided"}
                
                Create 3 different connector sentences that:
                1. Create curiosity and suspense
                2. Link back to the intro story if possible (use specific elements from the intro)
                3. Build anticipation for the payoff
                4. Use direct audience engagement
                5. Include transition words
                6. Are specific to the content being discussed
                7. Reference specific details from the sections
                
                Return only the 3 connector sentences, one per line, without numbering or explanations.
                Make them specific to the actual content, not generic.
                """
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.7
                )
                
                ai_suggestions = response.choices[0].message.content.strip().split('\n')
                for suggestion in ai_suggestions:
                    if "Connector:" in suggestion:
                        connector_text = suggestion.replace("Connector:", "").strip()
                        formatted_suggestion = f"Connector between Section {section_before.number} and {section_after.number}: \"{connector_text}\" (Place after line {section_before.end_line})"
                        suggestions.append(formatted_suggestion)
                
            except Exception as e:
                print(f"AI connector generation failed: {e}")
                # Fallback to generic suggestions
                suggestions.extend(self._create_fallback_connectors(section_before, section_after, intro, payoff))
        else:
            # Fallback to generic suggestions
            suggestions.extend(self._create_fallback_connectors(section_before, section_after, intro, payoff))
        
        return suggestions
    
    def _extract_section_content(self, script_text: str, section: ScriptSection) -> str:
        """Extract content around a section for AI analysis"""
        lines = script_text.split('\n')
        start_idx = max(0, section.start_line - 2)
        end_idx = min(len(lines), section.end_line + 2)
        
        content = '\n'.join(lines[start_idx:end_idx])
        return content[:500]  # Limit content length
    
    def _create_fallback_connectors(self, section_before: ScriptSection, section_after: ScriptSection, intro: str, payoff: str) -> List[str]:
        """Create fallback connector suggestions when AI is not available"""
        suggestions = []
        
        # Extract key themes from section titles and content
        before_theme = section_before.title.lower()
        after_theme = section_after.title.lower()
        before_content = section_before.content[:100].lower() if section_before.content else ""
        after_content = section_after.content[:100].lower() if section_after.content else ""
        
        # Generate contextual connector options based on actual content
        connector_options = []
        
        # Check for specific keywords in content to make suggestions more relevant
        if "price" in before_content or "cost" in before_content:
            connector_options.append("But here's what's really driving these price changes...")
            connector_options.append("Now, let's understand why these prices are actually rising...")
        elif "myth" in before_content or "believe" in before_content:
            connector_options.append("But here's the truth about this common misconception...")
            connector_options.append("Let me show you what's really happening here...")
        elif "market" in before_content or "invest" in before_content:
            connector_options.append("Now, here's what you need to know about the current market...")
            connector_options.append("This brings us to the most important part - how to actually benefit...")
        else:
            # Generic but still contextual suggestions
            connector_options.append(f"But here's what most people don't realize about {after_theme}...")
            connector_options.append(f"Now, let's look at what's actually happening with {after_theme}...")
        
        # Add intro-based suggestions if intro is provided
        if intro and len(intro) > 20:
            intro_keywords = intro.lower().split()[:5]  # Get first 5 words
            if any(keyword in before_content or keyword in after_content for keyword in intro_keywords):
                connector_options.append("Remember that story we started with? This is where it gets interesting...")
                connector_options.append("This connects back to what we discussed at the beginning...")
        
        # Format suggestions with specific placement info
        for connector_text in connector_options:
            suggestion = f"Connector between Section {section_before.number} and {section_after.number}: \"{connector_text}\" (Place after line {section_before.end_line})"
            suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_comprehensive_connector_suggestions(self, sections: List[ScriptSection], intro: str, payoff: str, script_text: str) -> List[str]:
        """Generate comprehensive connector suggestions when no connectors are found"""
        suggestions = []
        
        # Use AI to analyze the entire script and generate contextual connectors
        if self.openai_client:
            try:
                # Prepare comprehensive context
                sections_context = "\n".join([f"Section {s.number}: {s.title}\n{s.content[:200]}..." for s in sections])
                
                prompt = f"""
                Analyze this video script and create compelling connector sentences between each section.
                
                Script Context:
                {sections_context}
                
                Intro: {intro[:200] if intro else "No intro provided"}
                Payoff: {payoff[:200] if payoff else "No payoff provided"}
                
                Create 2-3 specific connector sentences for each section transition that:
                1. Create curiosity and suspense
                2. Link back to the intro story if possible
                3. Build anticipation for the payoff
                4. Use direct audience engagement
                5. Include transition words
                6. Are specific to the content being discussed
                
                Format each as: "Section X→Y: [connector sentence]"
                """
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=800,
                    temperature=0.7
                )
                
                ai_suggestions = response.choices[0].message.content.strip().split('\n')
                for suggestion in ai_suggestions:
                    if "Section" in suggestion and "→" in suggestion:
                        # Parse and format the suggestion
                        parts = suggestion.split(":", 1)
                        if len(parts) == 2:
                            section_info = parts[0].strip()
                            connector_text = parts[1].strip()
                            
                            # Extract section numbers
                            import re
                            match = re.search(r'Section (\d+)→(\d+)', section_info)
                            if match:
                                section_before = int(match.group(1))
                                section_after = int(match.group(2))
                                
                                # Find the corresponding section
                                before_section = next((s for s in sections if s.number == section_before), None)
                                if before_section:
                                    formatted_suggestion = f"Connector between Section {section_before} and {section_after}: \"{connector_text}\" (Place after line {before_section.end_line})"
                                    suggestions.append(formatted_suggestion)
                
            except Exception as e:
                print(f"AI comprehensive suggestion generation failed: {e}")
                # Fallback to basic suggestions
                suggestions.extend(self._generate_fallback_comprehensive_suggestions(sections, intro, payoff))
        else:
            # Fallback to basic suggestions
            suggestions.extend(self._generate_fallback_comprehensive_suggestions(sections, intro, payoff))
        
        return suggestions
    
    def _generate_fallback_comprehensive_suggestions(self, sections: List[ScriptSection], intro: str, payoff: str) -> List[str]:
        """Generate fallback comprehensive suggestions when AI is not available"""
        suggestions = []
        
        for i in range(len(sections) - 1):
            current_section = sections[i]
            next_section = sections[i + 1]
            
            # Extract key content from sections for better context
            current_content = current_section.content[:100].lower() if current_section.content else ""
            next_content = next_section.content[:100].lower() if next_section.content else ""
            
            # Generate contextual suggestions based on actual content
            connector_options = []
            
            # Check for specific keywords in content to make suggestions more relevant
            if "price" in current_content or "cost" in current_content:
                connector_options.append(f"But here's what's really driving these price changes...")
                connector_options.append(f"Now, let's understand why these prices are actually rising...")
            elif "myth" in current_content or "believe" in current_content:
                connector_options.append(f"But here's the truth about this common misconception...")
                connector_options.append(f"Let me show you what's really happening here...")
            elif "market" in current_content or "invest" in current_content:
                connector_options.append(f"Now, here's what you need to know about the current market...")
                connector_options.append(f"This brings us to the most important part - how to actually benefit...")
            else:
                # Generic but still contextual suggestions
                connector_options.append(f"But here's what most people don't realize about {next_section.title.lower()}...")
                connector_options.append(f"Now, let's look at what's actually happening with {next_section.title.lower()}...")
                connector_options.append(f"This leads us to the most important part - {next_section.title.lower()}...")
            
            # Add intro-based suggestions if intro is provided
            if intro and len(intro) > 20:
                intro_keywords = intro.lower().split()[:5]  # Get first 5 words
                if any(keyword in current_content or keyword in next_content for keyword in intro_keywords):
                    connector_options.append(f"Remember that story we started with? This is where it gets interesting...")
                    connector_options.append(f"This connects back to what we discussed at the beginning...")
            
            for connector_text in connector_options[:3]:  # Limit to 3 suggestions
                suggestion = f"Connector between Section {current_section.number} and {next_section.number}: \"{connector_text}\" (Place after line {current_section.end_line})"
                suggestions.append(suggestion)
        
        return suggestions
    
    def _generate_connector_improvements(self, connector: Connector, script_text: str, intro: str, payoff: str) -> List[str]:
        """Generate specific improvement suggestions for existing connectors using AI"""
        suggestions = []
        
        try:
            # Extract context around the connector
            lines = script_text.split('\n')
            start_idx = max(0, connector.line_number - 3)
            end_idx = min(len(lines), connector.line_number + 3)
            context = '\n'.join(lines[start_idx:end_idx])
            
            prompt = f"""
            Analyze this connector and provide specific improvement suggestions.
            
            Current connector: "{connector.text}"
            Issues: {', '.join(connector.issues)}
            Context: {context}
            Intro: {intro[:100] if intro else "No intro"}
            Payoff: {payoff[:100] if payoff else "No payoff"}
            
            Provide 2-3 specific improvement suggestions that:
            1. Address the identified issues
            2. Make the connector more engaging
            3. Better link to intro/payoff
            4. Include specific examples
            
            Format each as: "Improvement: [specific suggestion]"
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7
            )
            
            ai_suggestions = response.choices[0].message.content.strip().split('\n')
            for suggestion in ai_suggestions:
                if "Improvement:" in suggestion:
                    improvement_text = suggestion.replace("Improvement:", "").strip()
                    formatted_suggestion = f"Line {connector.line_number}: {improvement_text}"
                    suggestions.append(formatted_suggestion)
            
        except Exception as e:
            print(f"AI improvement generation failed: {e}")
            # Fallback to generic suggestions
            suggestions.append(f"Improve connector at line {connector.line_number}: {connector.text[:50]}... Issues: {', '.join(connector.issues)}")
        
        return suggestions
    
    def _analyze_connector(self, connector_text: str, sections: List[ScriptSection], line_number: int, section_before: Optional[int] = None, section_after: Optional[int] = None) -> Tuple[ConnectorType, bool, List[str]]:
        """Analyze a connector to determine its type and validity with improved logic"""
        issues = []
        connector_type = ConnectorType.SECTION_TO_SECTION
        
        # Check if it references intro
        intro_indicators = ['intro', 'introduction', 'hook', 'opening', 'start', 'beginning', 'first', 'initially']
        if any(keyword in connector_text.lower() for keyword in intro_indicators):
            connector_type = ConnectorType.INTRO
        
        # Check if it references payoff
        payoff_indicators = ['payoff', 'conclusion', 'ending', 'final', 'wrap up', 'closing', 'end', 'ultimately', 'finally']
        if any(keyword in connector_text.lower() for keyword in payoff_indicators):
            connector_type = ConnectorType.PAYOFF
        
        # Validate connector with more comprehensive checks
        is_valid = True
        
        # Check if connector is too short
        word_count = len(connector_text.split())
        if word_count < 8:
            issues.append("Connector is too short - needs more substance")
            is_valid = False
        elif word_count < 15:
            issues.append("Connector could be more detailed")
        
        # Check if connector has clear transition words
        transition_words = ['but', 'however', 'meanwhile', 'furthermore', 'moreover', 'now', 'that brings us to', 'speaking of', 'on that note', 'this leads us to', 'what\'s interesting', 'here\'s the thing']
        if not any(word in connector_text.lower() for word in transition_words):
            issues.append("Missing clear transition words")
            is_valid = False
        
        # Check if connector creates curiosity or suspense
        curiosity_words = ['but', 'however', 'what if', 'imagine', 'surprisingly', 'shockingly', 'here\'s the twist', 'plot twist', 'unexpected', 'stunning', 'amazing']
        if not any(word in connector_text.lower() for word in curiosity_words):
            issues.append("Doesn't create enough curiosity or suspense")
            is_valid = False
        
        # Check if connector links to intro or payoff
        if connector_type == ConnectorType.SECTION_TO_SECTION:
            issues.append("Doesn't clearly link back to intro or forward to payoff")
            is_valid = False
        
        # Check for engagement elements
        engagement_words = ['you', 'your', 'imagine', 'think about', 'consider', 'picture this', 'here\'s what', 'let me tell you']
        if not any(word in connector_text.lower() for word in engagement_words):
            issues.append("Missing direct audience engagement")
        
        return connector_type, is_valid, issues
    
    def _find_connector_sections(self, line_number: int, sections: List[ScriptSection]) -> Tuple[Optional[int], Optional[int]]:
        """Find which sections a connector bridges"""
        section_before = None
        section_after = None
        
        for i, section in enumerate(sections):
            if section.start_line <= line_number <= section.end_line:
                section_before = section.number
                if i + 1 < len(sections):
                    section_after = sections[i + 1].number
                break
        
        return section_before, section_after
    
    def _find_missing_connectors(self, sections: List[ScriptSection], connectors: List[Connector]) -> List[Tuple[int, int]]:
        """Find sections that need connectors"""
        missing = []
        
        # Check transitions between sections
        for i in range(len(sections) - 1):
            current_section = sections[i].number
            next_section = sections[i + 1].number
            
            # Check if there's a connector between these sections
            has_connector = any(
                c.section_before == current_section and c.section_after == next_section
                for c in connectors
            )
            
            if not has_connector:
                missing.append((current_section, next_section))
        
        return missing
    
    def _generate_suggestions(self, sections: List[ScriptSection], connectors: List[Connector], 
                            missing_connectors: List[Tuple[int, int]], intro: str, payoff: str, script_text: str) -> List[str]:
        """Generate suggestions for improving connectors"""
        suggestions = []
        
        # Generate specific connector suggestions for missing connectors
        specific_suggestions = self._generate_specific_connector_suggestions(sections, intro, payoff, script_text)
        suggestions.extend(specific_suggestions)
        
        # If no connectors found at all, generate comprehensive suggestions
        if not connectors:
            print("No connectors found, generating comprehensive suggestions...")
            print(f"Using intro for suggestions: {intro[:100] if intro else 'No intro available'}...")
            comprehensive_suggestions = self._generate_comprehensive_connector_suggestions(sections, intro, payoff, script_text)
            suggestions.extend(comprehensive_suggestions)
        
        # Analyze existing connectors and provide specific improvement suggestions
        for connector in connectors:
            if not connector.is_valid:
                # Use AI to generate specific improvement suggestions
                if self.openai_client:
                    try:
                        improvement_suggestions = self._generate_connector_improvements(connector, script_text, intro, payoff)
                        suggestions.extend(improvement_suggestions)
                    except Exception as e:
                        print(f"AI improvement generation failed: {e}")
                        # Fallback to generic suggestions
                        suggestions.append(f"Improve connector at line {connector.line_number}: {connector.text[:50]}... Issues: {', '.join(connector.issues)}")
                else:
                    # Fallback to generic suggestions
                    suggestions.append(f"Improve connector at line {connector.line_number}: {connector.text[:50]}... Issues: {', '.join(connector.issues)}")
            else:
                # Connector is valid, but we can still suggest enhancements
                suggestions.append(f"✓ Good connector at line {connector.line_number}: {connector.text[:50]}... (Consider adding more engagement)")
        
        # General suggestions
        if len(connectors) < len(sections) - 1:
            suggestions.append(f"Script has {len(sections)} sections but only {len(connectors)} connectors. Consider adding more transitions.")
        
        if not intro:
            suggestions.append("No clear intro found. Add a compelling opening hook that sets up the main story.")
        
        if not payoff:
            suggestions.append("No clear payoff found. Add a satisfying conclusion that ties back to the intro.")
        
        # Add strategic suggestions based on content analysis
        if sections:
            suggestions.append("Consider adding connectors that reference the opening story throughout the script to maintain viewer engagement.")
            suggestions.append("Use connectors to build anticipation for the final payoff or conclusion.")
            suggestions.append("Include personal anecdotes or relatable examples in connectors to increase engagement.")
        
        # Note: Specific connector suggestions are already generated above
        
        return suggestions
    
    def _calculate_score(self, sections: List[ScriptSection], connectors: List[Connector], 
                        missing_connectors: List[Tuple[int, int]]) -> float:
        """Calculate a score from 0-100 for the script's connector quality"""
        if not sections:
            return 0.0
        
        # Base score
        score = 100.0
        
        # Deduct for missing connectors
        expected_connectors = len(sections) - 1
        actual_connectors = len(connectors)
        missing_connector_penalty = (expected_connectors - actual_connectors) * 15
        score -= missing_connector_penalty
        
        # Deduct for invalid connectors
        invalid_connectors = sum(1 for c in connectors if not c.is_valid)
        invalid_connector_penalty = invalid_connectors * 10
        score -= invalid_connector_penalty
        
        # Bonus for intro/payoff connectors
        intro_payoff_connectors = sum(1 for c in connectors if c.type in [ConnectorType.INTRO, ConnectorType.PAYOFF])
        intro_payoff_bonus = intro_payoff_connectors * 5
        score += intro_payoff_bonus
        
        return max(0.0, min(100.0, score))
    
    def generate_connector_suggestions(self, section_before: ScriptSection, section_after: ScriptSection, 
                                     intro: str, payoff: str) -> List[str]:
        """Generate specific connector suggestions for a section transition"""
        suggestions = []
        
        # Use OpenAI for enhanced suggestions if available
        if self.openai_client:
            try:
                ai_suggestions = self._generate_ai_connector_suggestions(
                    section_before, section_after, intro, payoff
                )
                suggestions.extend(ai_suggestions)
            except Exception as e:
                print(f"OpenAI API error: {e}")
                # Fall back to rule-based suggestions
        
        # Rule-based fallback suggestions
        if not suggestions:
            # Intro-based connector
            if intro:
                intro_suggestion = f"Remember how we started with {intro[:100]}...? Well, that's exactly what's happening in {section_after.title.lower()}. "
                intro_suggestion += "But here's the twist that will change everything you thought you knew..."
                suggestions.append(intro_suggestion)
            
            # Payoff-based connector
            if payoff:
                payoff_suggestion = f"This brings us closer to understanding {payoff[:100]}... "
                payoff_suggestion += f"But before we get there, we need to explore {section_after.title.lower()}."
                suggestions.append(payoff_suggestion)
            
            # Curiosity-based connector
            curiosity_suggestion = f"You might think {section_before.title.lower()} was shocking, but wait until you hear about {section_after.title.lower()}. "
            curiosity_suggestion += "This is where the real story begins..."
            suggestions.append(curiosity_suggestion)
            
            # Contrast-based connector
            contrast_suggestion = f"While {section_before.title.lower()} showed us one side of the problem, {section_after.title.lower()} reveals something completely different. "
            contrast_suggestion += "And this is where it gets really interesting..."
            suggestions.append(contrast_suggestion)
        
        return suggestions
    
    def _generate_ai_connector_suggestions(self, section_before: ScriptSection, section_after: ScriptSection, 
                                         intro: str, payoff: str) -> List[str]:
        """Generate AI-powered connector suggestions using OpenAI"""
        
        prompt = f"""
You are a professional video script writer for Zero1 by Zerodha. Your job is to create compelling connectors between script sections that retain viewers by linking back to the intro or building anticipation for the payoff.

CONTEXT:
- Section {section_before.number}: {section_before.title}
- Section {section_after.number}: {section_after.title}
- Intro: {intro[:200] if intro else 'No intro provided'}
- Payoff: {payoff[:200] if payoff else 'No payoff provided'}

TASK:
Create 3-4 different connector suggestions that:
1. Smoothly transition from Section {section_before.number} to Section {section_after.number}
2. Either reference the intro story OR build anticipation for the payoff
3. Create curiosity and suspense to retain viewers
4. Are 1-2 sentences long
5. Use engaging, conversational tone
6. Include transition words like "but", "however", "meanwhile", "now", etc.

FORMAT:
Return only the connector suggestions, one per line, without numbering or bullet points.
"""
        
        response = self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional video script writer specializing in creating engaging connectors that retain viewers."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7
        )
        
        suggestions = response.choices[0].message.content.strip().split('\n')
        # Clean up suggestions
        suggestions = [s.strip() for s in suggestions if s.strip()]
        
        return suggestions[:4]  # Return max 4 suggestions

def main():
    """Main function for testing the bot"""
    bot = ScriptConnectorBot()
    
    # Sample script for testing
    sample_script = """
    Intro
    Today would have been my daughter's 18th birthday. But she won't be here to celebrate it.
    
    Section 1 - Licensing shortcuts
    First thing first, while researching the issue...
    
    Connecting line - And what happens when a country this populated hands out licenses like candy?
    
    Section 2 - Micro-Rule Breaks
    I was truly shocked at this number...
    
    Section 3 - Who Endangers Whom?
    You see, for 2W the moral is to protect riders...
    """
    
    analysis = bot.parse_script(sample_script)
    
    print("=== SCRIPT CONNECTOR ANALYSIS ===")
    print(f"Score: {analysis.score:.1f}/100")
    print(f"Sections: {len(analysis.sections)}")
    print(f"Connectors: {len(analysis.connectors)}")
    print(f"Missing connectors: {len(analysis.missing_connectors)}")
    print()
    
    print("=== SECTIONS ===")
    for section in analysis.sections:
        print(f"Section {section.number}: {section.title}")
    
    print()
    print("=== CONNECTORS ===")
    for connector in analysis.connectors:
        status = "✓" if connector.is_valid else "✗"
        print(f"{status} {connector.text[:100]}...")
        if connector.issues:
            print(f"  Issues: {', '.join(connector.issues)}")
    
    print()
    print("=== MISSING CONNECTORS ===")
    for before, after in analysis.missing_connectors:
        print(f"Need connector between Section {before} and Section {after}")
    
    print()
    print("=== SUGGESTIONS ===")
    for suggestion in analysis.suggestions:
        print(f"• {suggestion}")

if __name__ == "__main__":
    main()
