# -*- coding: utf-8 -*-
"""
سحب النصوص الحرفية للاجتماعات (Meeting Scripts) من موقع اللجنة.

Scraper for FOMC meeting transcripts.

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

# مكتبة 'Tika' تعتمد على إصدار 'Java'، لذا نستخدم 'textract' لأنّ ملف pdf نصّ بسيط أصلًا — Tika depends on the Java version, so use textract instead
# # User TIKA for pdf parsing
# os.environ['TIKA_SERVER_JAR'] = 'https://repo1.maven.org/maven2/org/apache/tika/tika-server/1.19/tika-server-1.19.jar'
# import tika
# from tika import parser
import textract

# استيراد الصنف الأب — Import parent class
from .FomcBase import FomcBase

class FomcMeetingScript(FomcBase):
    '''
    صنف عملي لاستخراج النصوص الحرفية لاجتماعات اللجنة (Meeting Scripts) من موقعها.
    تنشر اللجنة هذه النصوص بعد خمس سنوات، لذلك لا يمكن استخدامها للتنبؤ بالسياسة النقدية
    في الزمن الحقيقي.

    A convenient class for extracting meeting scripts from the FOMC website.
    FOMC publishes them after five years, so they cannot be used for real-time prediction.

    مثال للاستخدام — Example Usage:
        fomc = FomcMeetingScript()
        df = fomc.get_contents()
    '''
    def __init__(self, verbose = True, max_threads = 10, base_dir = '../data/FOMC/'):
        super().__init__('meeting_script', verbose, max_threads, base_dir)

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

        r = requests.get(self.calendar_url)
        soup = BeautifulSoup(r.text, 'html.parser')

        # النصوص الحرفية للاجتماعات متاحة في الأرشيف فقط لأنّها تُنشَر بعد خمس سنوات — Meeting Scripts are only in the archive, published after five years
        if from_year > 2014:
            print("Meeting scripts are available for 2014 or older")
        if from_year <= 2014:
            for year in range(from_year, 2015):
                yearly_contents = []
                fomc_yearly_url = self.base_url + '/monetarypolicy/fomchistorical' + str(year) + '.htm'
                r_year = requests.get(fomc_yearly_url)
                soup_yearly = BeautifulSoup(r_year.text, 'html.parser')
                meeting_scripts = soup_yearly.find_all('a', href=re.compile('^/monetarypolicy/files/FOMC\d{8}meeting.pdf'))
                for meeting_script in meeting_scripts:
                    self.links.append(meeting_script.attrs['href'])
                    self.speakers.append(self._speaker_from_date(self._date_from_link(meeting_script.attrs['href'])))
                    self.titles.append('FOMC Meeting Transcript')
                    self.dates.append(datetime.strptime(self._date_from_link(meeting_script.attrs['href']), '%Y-%m-%d'))
                if self.verbose: print("YEAR: {} - {} meeting scripts found.".format(year, len(meeting_scripts)))
            print("There are total ", len(self.links), ' links for ', self.content_type)

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

        link_url = self.base_url + link
        pdf_filepath = self.base_dir + 'script_pdf/FOMC_MeetingScript_' + self._date_from_link(link) + '.pdf'

        # النصوص متاحة بصيغة pdf فقط، فنحفظ الملف ونمرّر محتواه — Scripts are provided only in pdf; save it and pass the content
        res = requests.get(link_url)
        with open(pdf_filepath, 'wb') as f:
            f.write(res.content)

        # استخراج النصّ من ملف pdf — Extract text from the pdf
        # pdf_file_parsed = parser.from_file(pdf_filepath)
        # paragraphs = re.sub('(\n)(\n)+', '\n', pdf_file_parsed['content'].strip())
        pdf_file_parsed = textract.process(pdf_filepath).decode('utf-8')
        paragraphs = re.sub('(\n)(\n)+', '\n', pdf_file_parsed.strip())
        paragraphs = paragraphs.split('\n')

        section = -1
        paragraph_sections = []
        for paragraph in paragraphs:
            if not re.search('^(page|january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', paragraph.lower()):
                if len(re.findall(r'[A-Z]', paragraph[:10])) > 5 and not re.search('(present|frb/us|abs cdo|libor|rp–ioer|lsaps|cusip|nairu|s cpi|clos, r)', paragraph[:10].lower()):
                    section += 1
                    paragraph_sections.append("")
                if section >= 0:
                    paragraph_sections[section] += paragraph
        self.articles[index] = "\n\n[SECTION]\n\n".join([paragraph for paragraph in paragraph_sections])