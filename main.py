# -*- coding: utf-8 -*-
# Made By Code X Make on july 2023 
import os
import json
import re
import random
import string
import requests
import numpy as np
from random import randint
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageClip, CompositeVideoClip
import whisper_timestamped as whisper
import spacy
from keybert import KeyBERT
from googletrans import Translator as GoogleTranslator
from emoji_translate.emoji_translate import Translator as EmojiTranslator

# Configuration et initialisation des modèles
nlp_model_fr = spacy.load("fr_core_news_lg")
nlp_model_en = spacy.load("en_core_web_lg")
translator = GoogleTranslator()
emoji_translator = EmojiTranslator()

# Clé d'API Pexels
PEXELS_API = 'jdWJWr5H5vjUpoEwjO6nm4cc8Fe4sYdBQp29ky32xBU6SHUPeWPTWdMw'
headers = {'Authorization': PEXELS_API}


class VideoProcessor:
    
    def __init__(self, video_name):
        self.root_path = os.path.dirname(os.path.abspath(__file__))
        #On enleve l'extension
        video_name = video_name.split('.')[0]
        self.video_name = video_name

        self.video_path = os.path.join(self.root_path, "videos", video_name + ".mp4")
        self.audio_path = os.path.join(self.root_path, video_name, "audio.wav")
        self.transcription_file_path = os.path.join(self.root_path, video_name, "test.json")
        self.output_video_path = os.path.join(self.root_path, video_name, "after.mp4")
        self.config_file_path = os.path.join(self.root_path, video_name, "config.json")
        
        if not os.path.exists(os.path.join(self.root_path, video_name)):
            os.mkdir(os.path.join(self.root_path, video_name))

    def convert_video_to_audio(self):
        video = VideoFileClip(self.video_path)
        audio = video.audio
        audio.write_audiofile(self.audio_path)

    def load_audio_and_model(self):
        audio = whisper.load_audio(self.audio_path)
        model = whisper.load_model("small", device='cpu') #A modif plus tard pour un modéle plus puissant !
        return audio, model

    def transcribe_audio(self, model, audio):
        transcription = whisper.transcribe(model, audio, language="fr")
        for segment in transcription['segments']:
            if 'tokens' in segment:
                del segment['tokens']
        return transcription

    def write_transcription_to_file(self, transcription):
        with open(self.transcription_file_path, 'w', encoding='utf-8') as file:
            json.dump(transcription, file, ensure_ascii=False, indent=2)

    def extract_segments_from_json(self):
        with open(self.transcription_file_path, 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
            return data['segments']

    def decoupe_et_reorganise_texte(self, texte, taille_max=20):
        morceaux = []
        while len(texte) > 0:
            if len(texte) > taille_max:
                espace = texte.rfind(' ', 0, taille_max + 1)
                if espace == -1:
                    espace = taille_max
                morceau = texte[:espace]
                texte = texte[espace:].lstrip()

                while len(morceau) > taille_max:
                    sous_espace = morceau.rfind(' ', 0, taille_max + 1)
                    if sous_espace == -1:
                        sous_espace = taille_max
                    morceaux.append(morceau[:sous_espace])
                    morceau = morceau[sous_espace:].lstrip()

                morceaux.append(morceau)
            else:
                morceaux.append(texte)
                texte = ''

        for i in range(len(morceaux) - 1):
            mots = nlp_model_fr(morceaux[i])
            stopword_tail = ''

            for mot in reversed(mots):
                if mot.is_stop:
                    stopword_tail = mot.text_with_ws + stopword_tail
                else:
                    break

            if stopword_tail:
                morceaux[i] = morceaux[i][:len(morceaux[i]) - len(stopword_tail)]
                morceaux[i + 1] = stopword_tail + ' ' + morceaux[i + 1]

        return morceaux

    def config_into_two_lines(self, text, start, end):
        words = text.split(' ')
        if len(words) <= 1:
            return text, '', start, end

        middle = len(text) // 2
        split_index_before = text.rfind(' ', 0, middle)
        split_index_after = text.find(' ', middle)

        if abs(middle - split_index_before) < abs(middle - split_index_after):
            split_index = split_index_before
        else:
            split_index = split_index_after

        if split_index == -1:
            split_index = middle

        return text[:split_index], text[split_index:], start, end

    def search_broll(self, keyword, pageNumbers=1, resultsPerPage=20):
        translated_keyword = translator.translate(keyword, src='fr', dest='en').text
        print(translated_keyword)
        params = {
            'query': translated_keyword,
            'per_page': resultsPerPage,
            'page': pageNumbers,
            'orientation': 'portrait'
        }

        response = requests.get('https://api.pexels.com/videos/search', headers=headers, params=params)
        videos = response.json().get('videos', [])
        urls = [file.get('link') for video in videos for file in video.get('video_files', [])]
        print(f"Nombre de vidéos trouvées : {len(videos)}")

        if len(videos) == 0:
            print("Aucun résultat trouvé.")
        else:
            long_enough_videos = [video for video in videos if video.get('duration', 0) > 4]
            if not long_enough_videos:
                print("Aucune vidéo d'une durée supérieure à 4 secondes n'a été trouvée.")
                return None
            selected_video = random.choice(long_enough_videos)
            highest_quality_file = max(selected_video.get('video_files', []), key=lambda x: (x.get('quality'), x.get('height', 0)))
            url_video = highest_quality_file.get('link')
            print("URL de la vidéo sélectionnée :", url_video)
            response = requests.get(url_video)
            filename = os.path.join(self.root_path, "broll", f"video_{selected_video.get('id')}.mp4")
            print('FILENAME',filename)
            with open(filename, 'wb') as f:
                f.write(response.content)
            print(f"La vidéo a été téléchargée avec succès à l'emplacement suivant : {filename}")
            return filename

    def choice_emoji(self, text1, text2, start, end):
        if (end - start) < 0.5:
            return None
        text_complet = text1 + " " + text2
        print("Le texte complet est", text_complet)
        emoji = self.find_emoji(text_complet)
        print(emoji)
        return emoji

    def find_emoji(self, text):
            
        with open(os.path.join(self.root_path, "config", "emoji-FR.json"), "r", encoding="utf-8") as file:
            emoji_dict = json.load(file)

        punctuation_remover = str.maketrans("", "", string.punctuation.replace("'", ""))
        text = text.lower().replace("'", " '").translate(punctuation_remover).replace(" '", " ")
        doc_text = nlp_model_fr(text)
        found_emojis = []
        found_keywords = []

        for token_text in doc_text:
            if token_text.is_alpha and not token_text.is_stop and token_text.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'INTJ']:
                keyword = token_text.lemma_.lower()
                for emoji, keywords in emoji_dict.items():
                    if keyword in keywords:
                        found_emojis.append(emoji)
                        found_keywords.append(keyword)

        if found_emojis and found_keywords:
            print("Émojis trouvés :", found_emojis)
            print("Mots clés trouvés :", found_keywords)
            chosen_emoji = random.choice(found_emojis)
            chosen_keyword = random.choice(found_keywords)
            print(chosen_emoji, chosen_keyword)
            return chosen_emoji

        elif token_text.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'INTJ']:
            translated_text = translator.translate(text, src='fr', dest='en').text
            doc_translated_text = nlp_model_en(translated_text)
            print(translated_text)
            for token_text in doc_translated_text:
                if token_text.is_alpha and not token_text.is_stop and token_text.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV', 'INTJ']:
                    keyword = token_text.lemma_.lower()
                    emo = EmojiTranslator(exact_match_only=False, randomize=True)
                    chosen_emoji = emo.emojify(keyword)
                    print(chosen_emoji, keyword, "test")
                    if chosen_emoji == keyword:
                        print("Malheureusement pas d'emoji pour aujd !")
                        return None
                    else:
                        print(chosen_emoji)
                        return chosen_emoji

            print("Malheureusement pas d'emoji pour aujd !")
            return None

        print("Malheureusement pas d'emoji pour aujd !")
        return None

    def create_config(self, segments, taille_max=20):
        config = []
        config.append({"police": "", "font_size": [], "word_spacing": 0, "outline": [], "shadow": [], "writing_style": "uppercase only"})
        id_counter = 1
        for segment in segments:
            words = segment['words']
            chunks = self.decoupe_et_reorganise_texte(' '.join(word['text'] for word in words), taille_max=taille_max)
            word_index = 0
            print(segment, "segment")
            for chunk in chunks:
                chunk = chunk.replace(" ?", "?").replace(" !", "!").replace(" .", ".")
                sous_morceau_words = chunk.split()
                chunk_start_time = words[word_index]['start']
                word_index += len(sous_morceau_words)
                word_index = min(word_index, len(words))
                chunk_end_time = words[word_index - 1]['end']
                text1, text2, start, end = self.config_into_two_lines(chunk, chunk_start_time, chunk_end_time)
                if text1 and text2:
                    if text1[-1] in ['.', '!', '?'] and text2[0].isalpha():
                        text1 += ' '
                    elif text1[-1].isalpha() and text2[0] in ['.', '!', '?']:
                        text2 = ' ' + text2
                text1 = text1.strip()
                text2 = text2.strip()
                if text1 or text2:
                    print(chunk, "chunk")
                    print("Start time: ", start, "End time: ", end)
                    word_timestamps = [{"text": word["text"], "color": [], "start": word["start"], "end": word["end"]} for word in words[word_index - len(sous_morceau_words):word_index]]
                    emoji = self.choice_emoji(text1, text2, start, end)
                    #si l'emoji est none en ajoute ""
                    if emoji is None:
                        emoji = ""
                    config.append({'id': id_counter, 'emoji': emoji, 'animation_emoji': "", 'emoji_size': [], 'position_emoji': [], 'text1': text1, 'position_text1': [], 'text2': text2, 'position_text2': [], 'start': start, 'end': end, 'word_timestamps': word_timestamps})
                    id_counter += 1
        return config

    def write_config_to_file(self, index, rewrite=True, **kwargs):
        # Votre logique de réécriture ici
        directory = os.path.dirname(self.config_file_path)
        filename = f"config.json"
        filepath = os.path.join(directory, filename)
        print("filepath : ", filepath)

        if not rewrite:
            str_json = json.dumps(index, ensure_ascii=False, indent=2)
            str_json = re.sub(r'\[\s*([\d,\s]+?)\s*\]', lambda m: '[' + ' '.join(m.group(1).split()) + ']', str_json)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(str_json)
            return filepath

        with open(self.config_file_path, 'r', encoding='utf-8') as config_file:
            config = json.load(config_file)

        config[0]["police"] = kwargs.get('font_name')
        config[0]["font_size"] = [round(kwargs.get('font_size', 0), 1), round(kwargs.get('font_size_increase', 0), 1)]
        config[0]["word_spacing"] = kwargs.get('word_spacing')
        config[0]["outline"] = [kwargs.get('outline_color'), kwargs.get('outline_width')]
        config[0]["shadow"] = [kwargs.get('shadow_color'), kwargs.get('shadow_width'), kwargs.get('shadow_offset')[0], kwargs.get('shadow_offset')[1]]

        item = config[kwargs.get('id_text')]
        if kwargs.get('animation_name') is not None:
            item["animation_emoji"] = kwargs.get('animation_name')
            item["emoji_size"] = [kwargs.get('emoji_size')[0], kwargs.get('emoji_size')[1]]
            item["position_emoji"] = [kwargs.get('position_emoji')[0], kwargs.get('position_emoji')[1]]

        item["position_text1"] = [kwargs.get('x_av1'), kwargs.get('y_av1'), kwargs.get('x_ap1'), kwargs.get('y_av1')] #ATTENTION
        item["position_text2"] = [kwargs.get('x_av2'), kwargs.get('y_av2'), kwargs.get('x_ap2'), kwargs.get('y_av2')]
        # config.append({'id': id_counter, 'emoji': emoji, 'animation_emoji':"", 'emoji_size':[],'position_emoji':[], 'text1': text1, 'position_text1':[], 'text2': text2, 'position_text2':[], 'start': start, 'end': end, 'word_timestamps': word_timestamps})
        #            #INFO :                      emoji,         animation_1,        [x, y],              [x_min, y_min, x_max, y_max],              text1,      [x_av, y_av, x_ap, y_ap],          text2,      [x_av, y_av, x_ap, y_ap],         start,         end,                    word_timestamps        


        split_index = len(kwargs.get('word_colors', [])) // 2

        if len(kwargs.get('word_colors', [])) == 1:
            color1 = color2 = kwargs.get('word_colors', [])[0]
            if 0 < len(item['word_timestamps']):
                item['word_timestamps'][0]['color'] = [color1[1], color2[1]]
        else:
            word_colors_1 = kwargs.get('word_colors', [])[:split_index]
            word_colors_2 = kwargs.get('word_colors', [])[split_index:]
            for i, (color1, color2) in enumerate(zip(word_colors_1, word_colors_2)):
                if i < len(item['word_timestamps']):
                    item['word_timestamps'][i]['color'] = [color1[1], color2[1]]

        str_json = json.dumps(config[0:], ensure_ascii=False, indent=2)
        str_json = re.sub(r'\[\s*([-,\d,\s,"#a-fA-F]+?)\s*\]', lambda m: '[' + ' '.join(m.group(1).split()) + ']', str_json)
        str_json = re.sub(r'\[\n\s*([-,\d.,\s,"#a-fA-F]+?)\s*\]', lambda m: '[' + ', '.join(m.group(1).split()) + ']', str_json)
        str_json = re.sub(r'\[\n\s*([-,\d.,\s,"#a-fA-Fnull]+?)\s*\]', lambda m: '[' + ', '.join(m.group(1).split()) + ']', str_json)
        str_json = re.sub(r',+', ',', str_json)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(str_json)

        return filepath

    def generate_subtitles(self, video, create_subtitle_clip):
        subtitle_clips = []  # Liste pour stocker les clips de sous-titres
        emoji_clips = []  # Liste pour stocker les clips d'emojis
        b_roll_clips = []  # Liste pour stocker les clips B-roll

        # Charger la configuration des sous-titres à partir du fichier JSON
        with open(self.config_file_path, 'r', encoding='utf-8') as config:
            config = json.load(config)

        # Obtenir les temps de début pour le texte2 de chaque segment
        text2_start_times = self.get_text2_start_times(config)
        
        # Rechercher et sélectionner les mots-clés pour les clips B-roll
        selected_keywords = self.research_keyword()
        print("Mots-clés sélectionnés :", selected_keywords)

        # Pour chaque mot-clé sélectionné, rechercher et télécharger un clip B-roll correspondant
        for keyword, lemma, start, end in selected_keywords:
            b_roll_file_path = self.search_broll(lemma)
            if b_roll_file_path is not None:
                # Redimensionner et ajuster la durée du clip B-roll
                b_roll_clip = VideoFileClip(b_roll_file_path).without_audio().resize(video.size).subclip(0, randint(3, 4))
                b_roll_clips.append(b_roll_clip.set_start(start))

        # Pour chaque configuration de segment (excepté le premier qui contient les paramètres généraux)
        for i, config in enumerate(config[1:]):
            text1 = config['text1']  # Texte de la première ligne
            text2 = config['text2']  # Texte de la deuxième ligne
            start = config['start']  # Temps de début du segment
            end = config['end']  # Temps de fin du segment
            emoji = config['emoji']  # Emoji à afficher
            id_text = config['id']  # Identifiant du texte
            print(f'id_text: {id_text}, type: {type(id_text)}')
            text2_start_time = text2_start_times[i]  # Temps de début pour le texte2
            video_clip = video.subclip(start, end)  # Extraire le segment vidéo correspondant
            print(start)

            # Créer les clips de sous-titres et d'emojis pour ce segment
            txt_clip, emoji_clip, _ = create_subtitle_clip(video_clip, self.config_file_path, text1, text2, start, end, emoji, id_text, text2_start_time)

            # Ajouter le clip de sous-titres à la liste, en définissant son temps de début
            subtitle_clips.append(txt_clip.set_start(start))
            
            # Si un clip d'emoji est généré, l'ajouter à la liste avec son temps de début
            if emoji_clip is not None:
                emoji_clips.append(emoji_clip.set_start(start))

        # Retourner les listes de clips de sous-titres, d'emojis et de B-roll
        return subtitle_clips, emoji_clips, b_roll_clips

    def get_text2_start_times(self, config):
        text2_start_times = []
        for config in config[1:]:
            text1_words_count = len(config['text1'].split())
            word_timestamps = config['word_timestamps']
            if len(word_timestamps) > text1_words_count:
                text2_start_time = word_timestamps[text1_words_count]['start']
                text2_start_times.append(text2_start_time)
            else:
                text2_start_times.append(None)
        return text2_start_times

    def research_keyword(self):
        with open(self.transcription_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        full_text = data['text']
        doc = nlp_model_fr(full_text)
        sentences = [sent.text for sent in doc.sents if len(sent.text) > 40]
        kw_model = KeyBERT('distilbert-base-nli-mean-tokens')
        keywords_list = []
        blacklist = ['choses', 'faire', 'genre', 'oui', 'non']

        for sentence in sentences:
            sentence = "".join([token.text_with_ws for token in nlp_model_fr(sentence)])
            keywords = kw_model.extract_keywords(sentence, top_n=10)
            doc = nlp_model_fr(sentence)
            entities = [ent.text for ent in doc.ents if ent.label_ in ['MISC', 'LOC', 'PROPN', 'GPE']]
            for entity in entities:
                if entity.lower() not in blacklist:
                    keywords_list.append((entity, entity))
            for keyword, score in keywords:
                token = nlp_model_fr(keyword)[0]
                if not token.is_stop and token.pos_ in ['VERB', 'NOUN'] and keyword.lower() not in blacklist:
                    keywords_list.append((keyword, token.lemma_))

        keywords_timestamps = []
        keyword_counts = {keyword: 0 for keyword, _ in keywords_list}
        for keyword, lemma in keywords_list:
            count = keyword_counts[keyword]
            for segment in data['segments']:
                for i, word in enumerate(segment['words']):
                    if word['text'] == keyword:
                        if count == 0:
                            if word['start'] > 5 and word['end'] < data['segments'][-1]['end'] - 5:
                                keywords_timestamps.append((keyword, lemma, word['start'], word['end']))
                            break
                        else:
                            count -= 1
            keyword_counts[keyword] += 1

        keywords_timestamps.sort(key=lambda x: x[2])
        random.shuffle(keywords_timestamps)
        selected_keywords = []
        keyword_timestamps = {}
        current_time = 0
        tolerance = 5
        delay = randint(4, 8)

        for keyword, lemma, start, end in keywords_timestamps:
            if len(keyword_timestamps.get(keyword, [])) < 2:
                if start >= current_time + delay:
                    selected_keywords.append((keyword, lemma, start, end))
                    keyword_timestamps.setdefault(keyword, []).append((start, end))
                    current_time = end
                elif start >= current_time + delay - tolerance:
                    selected_keywords.append((keyword, lemma, start, end))
                    keyword_timestamps.setdefault(keyword, []).append((start, end))
                    current_time = end
            else:
                min_timestamp = min(keyword_timestamps[keyword], key=lambda x: x[0])
                if start > min_timestamp[0]:
                    selected_keywords.remove((keyword, lemma) + min_timestamp)
                    selected_keywords.append((keyword, lemma, start, end))
                    keyword_timestamps[keyword].remove(min_timestamp)
                    keyword_timestamps[keyword].append((start, end))

        entities_in_keywords = [keyword[0] for keyword in keywords_timestamps if keyword[0] in entities]
        if entities_in_keywords:
            while not any(keyword[0] in entities for keyword in selected_keywords):
                for keyword, lemma, start, end in keywords_timestamps:
                    if len(keyword_timestamps.get(keyword, [])) < 2:
                        if start >= current_time + delay and keyword in entities:
                            selected_keywords.append((keyword, lemma, start, end))
                            keyword_timestamps.setdefault(keyword, []).append((start, end))
                            current_time = end
                        elif start >= current_time + delay - tolerance and keyword in entities:
                            selected_keywords.append((keyword, lemma, start, end))
                            keyword_timestamps.setdefault(keyword, []).append((start, end))
                            current_time = end

        return selected_keywords

    def create_subtitle_clip(self, video_clip, config_file_path, text1, text2, start, end, emoji, id_text, text2_start_time=None, b_roll_clip=None):
        # Initialisation des variables
        txt_clip_first_half = None
        x_ap1 = None
        word_colors = []

        # Définir la résolution de référence
        reference_resolution = (1920, 1080)

        # Calculer le facteur de mise à l'échelle en fonction de la taille du clip vidéo
        scaling_factor = video_clip.size[1] / reference_resolution[1]

        # Définir la couleur du texte par défaut et choisir aléatoirement une couleur de remplissage parmi les options
        text_color = "#ffffff"
        colors = ["#fd0201", "#01fc26", "#f2e50e"]
        fill_color = random.choice(colors)  # Ici ça a choisi les couleurs au hasard pour la première génération !
        reference_font_size = 40
        font_size = int(reference_font_size * scaling_factor)
        font_size_increase = font_size * 1.1

        # Charger les polices avec les tailles calculées
        fonts = {
            'montserrat_black_1': ImageFont.truetype(os.path.join(self.root_path, "config", "montserrat-black.ttf"), font_size),
            'montserrat_black_2': ImageFont.truetype(os.path.join(self.root_path, "config", "montserrat-black.ttf"), font_size_increase),
        }

        # Sélectionner la police principale
        font = fonts['montserrat_black_1']
        font_name = next((name for name, value in fonts.items() if value is font), None)
        font_name = font_name.rstrip("_1")

        # Créer une image transparente de la taille du clip vidéo
        img = Image.new('RGBA', (video_clip.size[0], video_clip.size[1]), color=(0, 0, 0, 0))
        text1, text2 = text1.upper(), text2.upper()  # Convertir le texte en majuscules
        draw = ImageDraw.Draw(img)

        # Calculer la boîte de délimitation pour le texte1 et le texte2
        bbox1 = draw.textbbox((0, 0), text1, font=font)
        bbox2 = draw.textbbox((0, 0), text2, font=font)
        w1, h1 = bbox1[2] - bbox1[0], bbox1[3] - bbox1[1]
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]

        # Calculer les positions pour le texte
        y_av1 = 0.8 * video_clip.size[1]
        reference_spacing = reference_font_size + 6
        spacing = int(reference_spacing * (font_size / reference_font_size))
        y_av2 = y_av1 + spacing
        x_position1 = (video_clip.size[0] - w1) / 2
        x_av2 = (video_clip.size[0] - w2) / 2

        # Ajouter des ombres au texte pour le style
        shadow_img = Image.new('RGBA', img.size, (0, 0, 0, 0))
        shadow_draw = ImageDraw.Draw(shadow_img)
        shadow_color = "#000000E6"
        shadow_width = 25
        shadow_offset = (2, 2)
        outline_color = "#000000"
        outline_width = 7
        shadow_draw.text((x_position1 + shadow_offset[0], y_av1 + shadow_offset[1]), text1, font=font, fill=shadow_color, stroke_width=shadow_width, stroke_fill=outline_color)
        shadow_draw.text((x_av2 + shadow_offset[0], y_av2 + shadow_offset[1]), text2, font=font, fill=shadow_color, stroke_width=shadow_width, stroke_fill=outline_color)

        # Combiner les images d'ombres et appliquer un flou
        combined_img = Image.alpha_composite(img, shadow_img)
        combined_img_blurred = combined_img.filter(ImageFilter.GaussianBlur(radius=20))

        # Créer une image pour le texte1 avec les paramètres de style
        img_text1 = Image.new('RGBA', (video_clip.size[0], video_clip.size[1]), color=(0, 0, 0, 0))
        font_text1 = fonts['montserrat_black_2']
        draw_text1 = ImageDraw.Draw(img_text1)
        words = text1.split()
        total_width = 0
        word_widths = []
        word_spacing = int(10 * scaling_factor)

        # Calculer la largeur totale et les largeurs individuelles des mots pour centrer le texte
        for word in words:
            bbox_word = draw_text1.textbbox((0, 0), word, font=font_text1)
            w_word, h_word = bbox_word[2] - bbox_word[0], bbox_word[3] - bbox_word[1]
            total_width += w_word + word_spacing
            word_widths.append(w_word)
        total_width -= word_spacing
        x_av1 = (video_clip.size[0] - total_width) / 2
        encountered_non_stopword = False

        # Dessiner les mots du texte1, en changeant la couleur des mots non-stop
        for i, word in enumerate(words):
            if nlp_model_fr.vocab[word].is_stop and not encountered_non_stopword:
                fill_color_word = "#ffffff"
            else:
                fill_color_word = fill_color
                encountered_non_stopword = True
            draw_text1.text((x_av1, y_av1), word, font=font_text1, fill=fill_color_word, stroke_width=outline_width, stroke_fill=outline_color)
            x_av1 += word_widths[i] + word_spacing
            word_colors.append((word, fill_color_word))

        # Créer une image pour le texte2
        img_text2 = Image.new('RGBA', (video_clip.size[0], video_clip.size[1]), color=(0, 0, 0, 0))
        font_text2 = fonts['montserrat_black_1']
        draw_text2 = ImageDraw.Draw(img_text2)
        bbox2 = draw_text2.textbbox((0, 0), text2, font=font_text2)
        w2, h2 = bbox2[2] - bbox2[0], bbox2[3] - bbox2[1]
        x_av2 = (video_clip.size[0] - w2) / 2
        draw_text2.text((x_av2, y_av2), text2, font=font_text2, fill=text_color, stroke_width=outline_width, stroke_fill=outline_color)

        words = text2.split()
        for word in words:
            word_colors.append((word, text_color))

        # Combiner les images de texte et appliquer un flou
        combined_img_first_half = Image.alpha_composite(img_text1, img_text2)
        combined_img_first_half = Image.alpha_composite(combined_img_blurred, combined_img_first_half)
        frame_first_half = np.array(combined_img_first_half)
        clip_duration = end - start if text2_start_time is None else text2_start_time - start

        # Créer un clip vidéo pour la première partie du texte
        txt_clip_first_half = (ImageClip(frame_first_half)
                            .set_duration(clip_duration)
                            .set_start(start)
                            .set_position(('center', 'bottom')))

        # Si le texte2 commence après une certaine durée, créer un deuxième clip pour la suite du texte
        if text2_start_time is not None and (end - start) > 0.5 and text2_start_time < end:
            img_text1_second = Image.new('RGBA', (video_clip.size[0], video_clip.size[1]), color=(0, 0, 0, 0))
            font_text1_second = fonts['montserrat_black_1']
            draw_text1_second = ImageDraw.Draw(img_text1_second)
            bbox1_second = draw_text1_second.textbbox((0, 0), text1, font=font_text1_second)
            w1_second, h1_second = bbox1_second[2] - bbox1_second[0], bbox1_second[3] - bbox1_second[1]
            x_ap1 = (video_clip.size[0] - w1_second) / 2
            draw_text1_second.text((x_ap1, y_av1), text1, font=font_text1_second, fill=text_color, stroke_width=outline_width, stroke_fill=outline_color)

            words = text1.split()
            for word in words:
                word_colors.append((word, text_color))

            montserrat_black = fonts['montserrat_black_2']
            img_text2_second = Image.new('RGBA', (video_clip.size[0], video_clip.size[1]), color=(0, 0, 0, 0))
            font_text2_second = montserrat_black
            draw_text2_second = ImageDraw.Draw(img_text2_second)
            words = text2.split()
            total_width = 0
            word_widths = []

            for word in words:
                bbox_word = draw_text2_second.textbbox((0, 0), word, font=font_text2_second)
                w_word, h_word = bbox_word[2] - bbox_word[0], bbox_word[3] - bbox_word[1]
                total_width += w_word + word_spacing
                word_widths.append(w_word)
            total_width -= word_spacing
            x_ap2 = (video_clip.size[0] - total_width) / 2
            encountered_non_stopword = False

            for i, word in enumerate(words):
                if nlp_model_fr.vocab[word].is_stop and not encountered_non_stopword:
                    fill_color_word = "#ffffff"
                else:
                    fill_color_word = fill_color
                    encountered_non_stopword = True
                draw_text2_second.text((x_ap2, y_av2), word, font=font_text2_second, fill=fill_color_word, stroke_width=outline_width, stroke_fill=outline_color)
                x_ap2 += word_widths[i] + word_spacing
                word_colors.append((word, fill_color_word))

            # Combiner les images de la deuxième partie du texte
            combined_img_second_half = Image.alpha_composite(img_text1_second, img_text2_second)
            combined_img_second_half = Image.alpha_composite(combined_img_blurred, combined_img_second_half)
            frame_second_half = np.array(combined_img_second_half)

            # Créer un clip vidéo pour la deuxième partie du texte
            txt_clip_second_half = (ImageClip(frame_second_half)
                                    .set_duration(end - text2_start_time)
                                    .set_start(text2_start_time)
                                    .set_position(('center', 'bottom')))

            # Concatenner les deux clips de texte
            concatenated_clip = concatenate_videoclips([txt_clip_first_half, txt_clip_second_half])
            min_x_text2 = x_ap2 - total_width
            max_x_text2 = x_ap2

            # Créer un clip d'emoji
            emoji_clip, animation_name, emoji_size, position_emoji = self.create_emoji_clip(video_clip, start, end, text2_start_time, reference_resolution, y_av1, y_av2, w1, x_position1, w2, h1, emoji, min_x_text2, max_x_text2)
            
            # Écrire la configuration mise à jour dans le fichier
            self.write_config_to_file(0, rewrite=True, font_name=font_name, font_size=font_size, font_size_increase=font_size_increase, word_spacing=word_spacing, outline_color=outline_color, outline_width=outline_width, shadow_color=shadow_color, shadow_width=shadow_width, shadow_offset=shadow_offset, id_text=id_text, animation_name=animation_name, emoji_size=emoji_size, position_emoji=position_emoji, x_av1=x_av1, x_ap1=x_ap1, y_av1=y_av1, x_av2=x_av2, y_av2=y_av2, x_ap2=x_ap2, word_colors=word_colors)
            
            # Retourner les clips générés
            return concatenated_clip, emoji_clip, b_roll_clip
        else:
            min_x_text2 = 0
            max_x_text2 = 0
            x_ap2 = x_av2
            x_ap1 = x_av1 #SI le deuxiéme clip n'est pas créer, on a une valeur "classique"


            # Créer un clip d'emoji
            emoji_clip, animation_name, emoji_size, position_emoji = self.create_emoji_clip(video_clip, start, end, text2_start_time, reference_resolution, y_av1, y_av2, w1, x_position1, w2, h1, emoji, min_x_text2, max_x_text2)
            
            # Écrire la configuration mise à jour dans le fichier
            self.write_config_to_file(0, rewrite=True, font_name=font_name, font_size=font_size, font_size_increase=font_size_increase, word_spacing=word_spacing, outline_color=outline_color, outline_width=outline_width, shadow_color=shadow_color, shadow_width=shadow_width, shadow_offset=shadow_offset, id_text=id_text, animation_name=animation_name, emoji_size=emoji_size, position_emoji=position_emoji, x_av1=x_av1, x_ap1=x_ap1, y_av1=y_av1, x_av2=x_av2, y_av2=y_av2, x_ap2=x_ap2, word_colors=word_colors)
            
            # Retourner les clips générés
            return txt_clip_first_half, emoji_clip, b_roll_clip

    def create_emoji_clip(self, video_clip, start, end, text2_start_time, reference_resolution, y_av1, y_av2, w1, x_position1, w2, h1, emoji, min_x_text2, max_x_text2):

        filename = os.path.join(self.root_path, "config", "emoji_paths.json")
        print(filename)

        with open(os.path.join(self.root_path, "config", "emoji_paths.json"), 'r', encoding='utf-8') as f:
            emoji_paths = json.load(f)
            
        try:
            emoji_file_path = os.path.join(self.root_path, "config", f"{emoji_paths[emoji]}")
        except KeyError:
            return None, None, None, (None, None)

        if (end - start) < 0.5:
            return None, None, None, (None, None)

        emoji_img = Image.open(emoji_file_path).convert("RGBA")
        emoji_scaling_factor = video_clip.size[1] / reference_resolution[1]
        emoji_scale = 0.5
        emoji_size = tuple([int(x * emoji_scaling_factor * emoji_scale) for x in emoji_img.size])
        emoji_img = emoji_img.resize(emoji_size, Image.LANCZOS)

        emoji_img_canvas = Image.new('RGBA', video_clip.size, (0, 0, 0, 0))
        x_av1 = video_clip.size[0] / 2 - emoji_size[0] / 2
        positions = [
            (x_av1, y_av1 - emoji_size[1]),
            (x_av1, y_av2 + emoji_size[1] - 50),
            (x_position1 - emoji_size[0] - 10, y_av1 - h1 - 50),
            (x_position1 + w1, y_av1 - h1 - 50)
        ]
        x_av1, y_ap1 = random.choice(positions)
        x_av1, y_ap1 = random.choice(positions)
        emoji_img_canvas.paste(emoji_img, (int(x_av1), int(y_ap1)), emoji_img)
        frame = np.array(emoji_img_canvas)

        normal = (ImageClip(frame)
                  .set_duration(end - start)
                  .set_start(start)
                  .set_position(('center', 'bottom')))

        min_y_1 = y_av1
        max_y_1 = y_av1 - emoji_size[1]
        speed_1 = -1300
        time_to_reach_max_y_1 = (max_y_1 - min_y_1) / speed_1

        if time_to_reach_max_y_1 > video_clip.duration:
            speed_1 = (max_y_1 - min_y_1) / video_clip.duration

        animation_1 = (ImageClip(np.array(emoji_img))
                       .set_duration(video_clip.duration)
                       .set_position(lambda t: (x_av1, min_y_1 + speed_1 * t if min_y_1 + speed_1 * t >= max_y_1 else max_y_1)))

        min_y_2 = y_av2 - 50
        max_y_2 = y_av2 + emoji_size[1] - 50
        speed_2 = 1300
        time_to_reach_max_y_2 = (max_y_2 - min_y_2) / speed_2

        if time_to_reach_max_y_2 > video_clip.duration:
            speed_2 = (max_y_2 - min_y_2) / video_clip.duration

        animation_2 = (ImageClip(np.array(emoji_img))
                       .set_duration(video_clip.duration)
                       .set_position(lambda t: (x_av1, min_y_2 + speed_2 * t if min_y_2 + speed_2 * t <= max_y_2 else max_y_2)))

        min_size_3 = 0.4
        max_size_3 = 1.0
        zoom_speed = 6
        time_to_reach_max_size_3 = (max_size_3 - min_size_3) / zoom_speed

        if time_to_reach_max_size_3 > video_clip.duration:
            zoom_speed = (max_size_3 - min_size_3) / video_clip.duration

        animation_3 = (ImageClip(np.array(emoji_img))
                       .set_duration(video_clip.duration)
                       .set_position(lambda t: (x_av1 + (emoji_size[0] - (min_size_3 + zoom_speed * t if min_size_3 + zoom_speed * t <= max_size_3 else max_size_3) * emoji_size[0]) / 2,
                                                y_ap1 + (emoji_size[1] - (min_size_3 + zoom_speed * t if min_size_3 + zoom_speed * t <= max_size_3 else max_size_3) * emoji_size[1]) / 2))
                       .resize(lambda t: ((min_size_3 + zoom_speed * t if min_size_3 + zoom_speed * t <= max_size_3 else max_size_3) * emoji_size[0],
                                          (min_size_3 + zoom_speed * t if min_size_3 + zoom_speed * t <= max_size_3 else max_size_3) * emoji_size[1])))

        min_x = min_x_text2 - emoji_size[0] / 2
        max_x = max_x_text2 - emoji_size[0]
        speed_x = 1500
        time_to_reach_max_x = (max_x - min_x) / speed_x

        if time_to_reach_max_x > video_clip.duration:
            speed_x = (max_x - min_x) / video_clip.duration

        animation_4 = (ImageClip(np.array(emoji_img))
                       .set_duration(video_clip.duration)
                       .set_position(lambda t: (min_x + speed_x * t if min_x + speed_x * t <= max_x else max_x, y_av2 + emoji_size[1] - 60)))

        animations_1 = {
            "normal": normal,
            "animation_3": animation_3
        }

        animations_2 = {
            "animation_1": animation_1,
            "animation_2": animation_2,
            "normal": normal
        }

        if text2_start_time is not None:
            animations_2["animation_4"] = animation_4

        if (x_av1, y_ap1) == (x_position1 - emoji_size[0] - 10, y_av1 - h1 - 50) or (x_av1, y_ap1) == (x_position1 + w1, y_av1 - h1 - 50):
            key, emoji_clip = random.choice(list(animations_1.items()))
        else:
            key, emoji_clip = random.choice(list(animations_2.items()))

        return emoji_clip, key, emoji_size, (x_av1, y_ap1)

    def write_final_video(self, video_path, subtitle_clips, emoji_clips, b_roll_clips, output_video_path):
        video = VideoFileClip(video_path)
        
        # Créez une vidéo pour les clips d'emoji en les combinant
        emoji_video = CompositeVideoClip(emoji_clips, size=video.size)
        
        # Créez une vidéo pour les clips de sous-titres en les combinant
        subtitle_video = CompositeVideoClip(subtitle_clips, size=video.size)
        
        # Combinez les vidéos (vidéo originale, clips B-roll, vidéo d'emoji et vidéo de sous-titres) dans l'ordre souhaité
        final_video = CompositeVideoClip([video] + b_roll_clips + [emoji_video] + [subtitle_video])
        
        subtitle_and_emoji_video = CompositeVideoClip([emoji_video] + [subtitle_video])

        # Vérifiez si le répertoire de sortie existe, sinon créez-le
        output_dir = os.path.dirname(output_video_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Écrivez la vidéo finale sur le disque
        final_video.write_videofile(output_video_path, codec="libx264", audio_codec='aac', threads=12, fps=24)

        # Retirer le nom de la vidéo de base : 
        output_subtitle_video_path = output_video_path.replace(".mp4", "_subtitle.mp4")
        
        # Vérifiez si le répertoire de sortie existe, sinon créez-le
        output_subtitle_dir = os.path.dirname(output_subtitle_video_path)
        if not os.path.exists(output_subtitle_dir):
            os.makedirs(output_subtitle_dir)
        
        subtitle_and_emoji_video.write_videofile(output_subtitle_video_path, codec="libx264", audio_codec='aac', threads=12, fps=24)


    def create_first_time_clip(self):
        self.convert_video_to_audio()
        audio, model = self.load_audio_and_model()
        transcription = self.transcribe_audio(model, audio)
        self.write_transcription_to_file(transcription)

        segments = self.extract_segments_from_json()
        config = self.create_config(segments)
        self.write_config_to_file(config, rewrite=False)

        video = VideoFileClip(self.video_path)
        subtitle_clips, emoji_clips, b_roll_clips = self.generate_subtitles(video, self.create_subtitle_clip)
        self.write_final_video(self.video_path, subtitle_clips, emoji_clips, b_roll_clips, self.output_video_path)
        
        #PARTIE 2  GEN 2
        
    #PARTIE GEN 2    
    #FONCTIONNE TEST V2
    """
    def create_subtitle_clip_from_config(self, video_clip, config):
        # Extraire les données de configuration avec des valeurs par défaut
        text1 = config.get('text1', "")
        text2 = config.get('text2', "")
        start = config.get('start', 0)
        end = config.get('end', 0)
        emoji = config.get('emoji', None)
        x_position1 = config.get('position_text1', [0, 0, None, None])[0]
        y_av1 = config.get('position_text1', [0, 0, None, None])[1]
        x_av2 = config.get('position_text2', [0, 0, None, None])[0]
        y_av2 = config.get('position_text2', [0, 0, None, None])[1]
        x_ap1 = config.get('position_text1', [0, 0, None, None])[2]
        y_position1_second = config.get('position_text1', [0, 0, None, None])[3]
        x_ap2 = config.get('position_text2', [0, 0, None, None])[2]
        y_position2_second = config.get('position_text2', [0, 0, None, None])[3]

        word_colors = [wt.get('color', ["#ffffff", "#ffffff"]) for wt in config.get('word_timestamps', [])]

        # Définir les valeurs par défaut pour les paramètres de style
        font_name = config.get('police', 'montserrat-black')
        font_size = int(config.get('font_size', [40])[0])
        outline_color = config.get('outline', ["#000000"])[0]
        outline_width = config.get('outline', [0, 7])[1]

        # Chemin du fichier de police
        font_path = os.path.join(self.root_path, "config", f"{font_name}.ttf")

        # Vérifier si le fichier de police existe
        if not os.path.exists(font_path):
            raise FileNotFoundError(f"Le fichier de police {font_path} est introuvable.")

        # Charger la police
        font = ImageFont.truetype(font_path, font_size)

        # Créer une image transparente de la taille du clip vidéo
        img = Image.new('RGBA', (video_clip.size[0], video_clip.size[1]), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        # Dessiner le texte1
        draw.text((x_position1, y_av1), text1, font=font, fill=word_colors[0][0], stroke_width=outline_width, stroke_fill=outline_color)
        if x_ap1 is not None:
            draw.text((x_ap1, y_position1_second), text1, font=font, fill=word_colors[0][1], stroke_width=outline_width, stroke_fill=outline_color)

        # Dessiner le texte2
        draw.text((x_av2, y_av2), text2, font=font, fill=word_colors[-1][0], stroke_width=outline_width, stroke_fill=outline_color)
        if x_ap2 is not None:
            draw.text((x_ap2, y_position2_second), text2, font=font, fill=word_colors[-1][1], stroke_width=outline_width, stroke_fill=outline_color)

        # Convertir l'image en clip vidéo
        frame = np.array(img)
        txt_clip = (ImageClip(frame)
                    .set_duration(end - start)
                    .set_start(start)
                    .set_position(('center', 'bottom')))

        # Créer le clip d'emoji s'il y en a un
        emoji_clip = None
        if emoji:
            emoji_clip = self.create_emoji_clip_from_config(video_clip, config)

        return txt_clip, emoji_clip, None  # Pas de clip B-roll dans cette fonction

    def create_emoji_clip_from_config(self, video_clip, config):
        emoji = config.get('emoji', None)
        emoji_size = config.get('emoji_size', [100, 100])
        position_emoji = config.get('position_emoji', [0, 0])
        start = config.get('start', 0)
        end = config.get('end', 0)

        if not emoji:
            return None

        # Charger l'emoji
        emoji_file_path = os.path.join(self.root_path, "config", "emojis", f"{emoji}.png")
        if not os.path.exists(emoji_file_path):
            raise FileNotFoundError(f"Le fichier d'emoji {emoji_file_path} est introuvable.")

        emoji_img = Image.open(emoji_file_path).convert("RGBA")
        emoji_img = emoji_img.resize((emoji_size[0], emoji_size[1]), Image.LANCZOS)

        # Créer une image transparente de la taille du clip vidéo
        img = Image.new('RGBA', (video_clip.size[0], video_clip.size[1]), color=(0, 0, 0, 0))
        img.paste(emoji_img, (int(position_emoji[0]), int(position_emoji[1])), emoji_img)
        
        frame = np.array(img)
        emoji_clip = (ImageClip(frame)
                    .set_duration(end - start)
                    .set_start(start)
                    .set_position(('center', 'bottom')))

        return emoji_clip

    def generate_subtitles_from_config(self, video):
        subtitle_clips = []
        emoji_clips = []
        b_roll_clips = []

        with open(self.config_file_path, 'r', encoding='utf-8') as config_file:
            try:
                config_data = json.load(config_file)
            except json.JSONDecodeError as e:
                print(f"Erreur de décodage JSON: {e}")
                print("Vérifiez la syntaxe du fichier JSON.")
                raise

        for config in config_data[1:]:  # Ignorer la première entrée qui contient les paramètres généraux
            start = config.get('start', 0)
            end = config.get('end', 0)
            video_clip = video.subclip(start, end)
            
            txt_clip, emoji_clip, _ = self.create_subtitle_clip_from_config(video_clip, config)
            
            subtitle_clips.append(txt_clip.set_start(start))
            if emoji_clip is not None:
                emoji_clips.append(emoji_clip.set_start(start))

        return subtitle_clips, emoji_clips, b_roll_clips

    def create_clip_with_config(self):
        video = VideoFileClip(self.video_path)
        subtitle_clips, emoji_clips, b_roll_clips = self.generate_subtitles_from_config(video)

        self.write_final_video(self.video_path, subtitle_clips, emoji_clips, b_roll_clips, self.output_video_path)
        """


# TEST
processor = VideoProcessor("26.mp4")
processor.create_first_time_clip()
