# -*- coding: utf-8 -*-
"""
سحب خطابات أعضاء اللجنة (Speeches) من موقع اللجنة.

Scraper for FOMC speeches.

المؤلف / Author: Dr Merwan Roudane
المستودع / Repository: https://github.com/merwanroudane/nlpfullarabic
"""

from datetime import datetime
import threading
import sys
import os
import pickle
import re

import requests
from bs4 import BeautifulSoup

import numpy as np
import pandas as pd

# استيراد الصنف الأب — Import parent class
from .FomcBase import FomcBase

class FomcSpeech(FomcBase):
    '''
    صنف عملي لاستخراج الخطابات (Speeches) من موقع اللجنة.

    A convenient class for extracting speeches from the FOMC website.

    مثال للاستخدام — Example Usage:
        fomc = FomcSpeech()
        df = fomc.get_contents()
    '''
    def __init__(self, verbose = True, max_threads = 10, base_dir = '../data/FOMC/'):
        super().__init__('speech', verbose, max_threads, base_dir)
        self.speech_base_url = self.base_url + '/newsevents/speech'

    def _get_links(self, from_year):
        '''
        إعادة تعريف دالة خاصة تُحدّد كل روابط المحتويات المطلوب تنزيلها من موقع اللجنة،
        من from_year (=min(2015, from_year)) إلى أحدث سنة متاحة.

        Override of the private function that sets all the links for the contents to
        download from the FOMC website.
        '''
        self.links = []
        self.titles = []
        self.speakers = []
        self.dates = []

        res = requests.get(self.calendar_url)
        soup = BeautifulSoup(res.text, 'html.parser')

        if self.verbose: print("Getting links for speeches...")
        to_year = datetime.today().strftime("%Y")

        if from_year <= 1995:
            print("Archive only from 1996, so setting from_year as 1996...")
            from_year = 1996
        for year in range(from_year, int(to_year)+1):
            # المؤرشَف بين 1996 و2005، وقد تغيّر الرابط منذ 2011 — Archived between 1996 and 2005; URL changed from 2011
            if year < 2011:
                speech_url = self.speech_base_url + '/' + str(year) + 'speech.htm'
            else:
                speech_url = self.speech_base_url + '/' + str(year) + '-speeches.htm'

            res = requests.get(speech_url)
            soup = BeautifulSoup(res.text, 'html.parser')
            speech_links = soup.findAll('a', href=re.compile('^/?newsevents/speech/.*{}\d\d\d\d.*.htm|^/boarddocs/speeches/{}/|^{}\d\d\d\d.*.htm'.format(str(year), str(year), str(year))))
            for speech_link in speech_links:
                # يُستخدَم الرابط نفسه أحيانًا لمشاهدة الفيديو المباشر، فنتجاوزه — The same link is sometimes used for live video; skip those
                if speech_link.find({'class': 'watchLive'}):
                    continue

                # إضافة الرابط والعنوان والتاريخ — Add link, title and date
                self.links.append(speech_link.attrs['href'])
                self.titles.append(speech_link.get_text())
                self.dates.append(datetime.strptime(self._date_from_link(speech_link.attrs['href']), '%Y-%m-%d'))

                # إضافة المتحدّث — Add speaker
                # في صفحة 1997 وحدها يأتي المتحدّث قبل الرابط، بخلاف بقية الصفحات — In the 1997 page only, the speaker precedes the link
                if year == 1997:
                    # في صفحة 1997، رابط خطاب 15 ديسمبر وحده يأتي المتحدّث بعده — In the 1997 page, only the December 15 speech has the speaker after the link
                    if speech_link.get('href') == '/boarddocs/speeches/1997/19971215.htm':
                        tmp_speaker = speech_link.parent.next_sibling.next_element.get_text().replace('\n', '').strip()
                    else:
                        tmp_speaker = speech_link.parent.previous_sibling.previous_sibling.get_text().replace('\n', '').strip()
                else:
                    # التاريخان 20051128 و20051129 مبنيّان بشكل مختلف — 20051128 and 20051129 are structured differently
                    if speech_link.get('href') in ('/boarddocs/speeches/2005/20051128/default.htm', '/boarddocs/speeches/2005/20051129/default.htm'):
                        tmp_speaker = speech_link.parent.previous_sibling.previous_sibling.get_text().replace('\n', '').strip()
                    tmp_speaker = speech_link.parent.next_sibling.next_element.get_text().replace('\n', '').strip()
                    # عندما تُوضَع أيقونة فيديو بين الرابط والمتحدّث — When a video icon is placed between the link and speaker
                    if tmp_speaker in ('Watch Live', 'Video'):
                        tmp_speaker = speech_link.parent.next_sibling.next_sibling.next_sibling.next_element.get_text().replace('\n', '').strip()
                self.speakers.append(tmp_speaker)
            if self.verbose: print("YEAR: {} - {} speeches found.".format(year, len(speech_links)))

    def _add_article(self, link, index=None):
        '''
        إعادة تعريف دالة خاصة تُضيف المقال المرتبط برابط واحد إلى متغيّر النسخة.
        المُعامل index هو موضع الإضافة، وبسبب المعالجة المتوازية يجب ضمان الترتيب الصحيح.

        Override of the private function that adds a related article for one link into
        the instance variable, preserving order under concurrent processing.
        '''
        if self.verbose:
            sys.stdout.write(".")
            sys.stdout.flush()

        res = requests.get(self.base_url + link)
        html = res.text
        # الوسم p غير مُغلَق بشكل سليم في حالات كثيرة — the p tag is not properly closed in many cases
        html = html.replace('<P', '<p').replace('</P>', '</p>')
        html = html.replace('<p', '</p><p').replace('</p><p', '<p', 1)
        # إزالة كل ما يأتي بعد الملحق أو المراجع — remove all after appendix or references
        x = re.search(r'(<b>references|<b>appendix|<strong>references|<strong>appendix)', html.lower())
        if x:
            html = html[:x.start()]
            html += '</body></html>'
        # تحليل نصّ html بواسطة 'BeautifulSoup' — Parse html text with BeautifulSoup
        article = BeautifulSoup(html, 'html.parser')
        # إزالة الحواشي — Remove footnote
        for fn in article.find_all('a', {'name': re.compile('fn\d')}):
            if fn.parent:
                fn.parent.decompose()
            else:
                fn.decompose()
        # جلب كل وسوم p — Get all p tags
        paragraphs = article.findAll('p')
        self.articles[index] = "\n\n[SECTION]\n\n".join([paragraph.get_text().strip() for paragraph in paragraphs])