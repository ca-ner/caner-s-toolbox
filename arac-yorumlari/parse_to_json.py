#!/usr/bin/env python3
"""
HTML tablo → JSON dönüştürücü
Kullanım: python3 parse_table.py input.html output.json
"""

import sys
import json
from html.parser import HTMLParser


class TableParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.records = []
        self.current_row = []
        self.current_text = ""
        self.link_href = ""
        self.in_td = False
        self.in_link = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "tr":
            self.current_row = []
        elif tag == "td":
            self.in_td = True
            self.current_text = ""
            self.link_href = ""
        elif tag == "a" and self.in_td:
            self.in_link = True
            self.link_href = attrs.get("href", "")

    def handle_endtag(self, tag):
        if tag == "td":
            self.in_td = False
            self.current_row.append({
                "text": self.current_text.strip(),
                "link": self.link_href or None
            })
        elif tag == "a":
            self.in_link = False
        elif tag == "tr":
            if len(self.current_row) == 6:
                try:
                    record = {
                        "sira":  int(self.current_row[0]["text"]),
                        "marka": self.current_row[1]["text"],
                        "model": self.current_row[2]["text"],
                        "yil":   self.current_row[3]["text"],
                        "yorum": self.current_row[4]["text"],
                        "video": {
                            "ad":  self.current_row[5]["text"],
                            "url": self.current_row[5]["link"]
                        }
                    }
                    self.records.append(record)
                except (ValueError, IndexError) as e:
                    print(f"[UYARI] Satır atlandı: {e}", file=sys.stderr)

    def handle_data(self, data):
        if self.in_td:
            self.current_text += data


def parse(html: str) -> list:
    parser = TableParser()
    parser.feed(html)
    return parser.records


if __name__ == "__main__":
    # Komut satırı argümanları
    input_file  = sys.argv[1] if len(sys.argv) > 1 else None
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    if input_file:
        with open(input_file, encoding="utf-8") as f:
            html = f.read()
    else:
        print("HTML dosyası belirtilmedi, stdin'den okunuyor...", file=sys.stderr)
        html = sys.stdin.read()

    records = parse(html)

    result_json = json.dumps(records, ensure_ascii=False, indent=2)

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result_json)
        print(f"{len(records)} kayıt → {output_file}", file=sys.stderr)
    else:
        print(result_json)
