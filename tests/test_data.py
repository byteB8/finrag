import unittest

from core import data


class Chunking(unittest.TestCase):
    def test_short_page_is_one_chunk(self):
        self.assertEqual(data.chunk_page("a b c\nd e", max_size=10, overlap=2), ["a b c\nd e"])

    def test_chunks_respect_size_and_overlap(self):
        text = "\n".join(f"w{i} x y" for i in range(10))          # 10 lines of 3 words
        chunks = data.chunk_page(text, max_size=9, overlap=3)
        for c in chunks:
            self.assertLessEqual(len(c.split()), 9)
        # each chunk after the first starts with the previous chunk's last line
        for prev, cur in zip(chunks, chunks[1:]):
            self.assertEqual(cur.splitlines()[0], prev.splitlines()[-1])
        # every line appears somewhere, in order
        joined = [line for c in chunks for line in c.splitlines()]
        self.assertEqual(sorted(set(joined), key=lambda s: int(s.split()[0][1:])), text.splitlines())

    def test_oversized_line_is_its_own_chunk(self):
        chunks = data.chunk_lines(["a", "big", "b"], [1, 50, 1], max_size=10, overlap=0)
        self.assertEqual(chunks, ["a", "big", "b"])

    def test_long_line_is_cut_into_word_windows(self):
        lines = data.page_lines(" ".join(str(i) for i in range(250)), max_words=120)
        self.assertEqual([len(line.split()) for line in lines], [120, 120, 10])

    def test_custom_sizes(self):
        chunks = data.chunk_page("1 2 3\nfour", max_size=4, overlap=0, sizes_of=lambda ls: [len(s) for s in ls])
        self.assertEqual(chunks, ["1 2 3", "four"])

    def test_clean_page(self):
        self.assertEqual(data.clean_page("  a  b \n\n\t c  \n"), "a b\nc")


class Entities(unittest.TestCase):
    def test_company_aliases(self):
        cases = {
            "Does AMEX have an improving operating margin profile as of 2022?": ["American Express"],
            "Are JnJ's FY2022 financials that of a high growth company?": ["Johnson & Johnson"],
            "What is the FY2017 return on assets (ROA) for Coca Cola?": ["Coca-Cola"],
            "What is the FY2018 capital expenditure amount for 3M?": ["3M"],
            "Which of JPM's business segments had the lowest net revenue in 2021 Q1?": ["JPMorgan"],
            "Which region had the Highest EBITDAR Contribution for MGM during FY2022?": ["MGM Resorts"],
        }
        for question, expected in cases.items():
            self.assertEqual(data.detect_companies(question), expected, question)

    def test_no_false_hits_on_ordinary_words(self):
        self.assertEqual(data.detect_companies("Was there a block of shares sold to a cost centre?"), [])
        self.assertEqual(data.detect_companies("Is the BAL sheet strong?"), [])

    def test_years(self):
        self.assertEqual(data.detect_years("Between FY2021 and FY 2022, and in Q2 of FY2024"), ["2021", "2022", "2024"])
        self.assertEqual(data.detect_years("No year 20219 here"), [])


class Unanswerable(unittest.TestCase):
    qs = [{"financebench_id": f"financebench_id_{i:05d}", "question": q} for i, q in enumerate([
        "What is the FY2018 capital expenditure for 3M?",
        "Does AMEX have an improving operating margin profile as of 2022?",
        "Are JnJ's financials those of a high growth company? Is JnJ growing?",
        "What drove the reduction in SG&A expense?"])]

    def test_years_move_and_companies_swap(self):
        out = data.make_unanswerable(self.qs, n_each=2, seed=0)
        years = [o for o in out if o["kind"] == "year"]
        comps = [o for o in out if o["kind"] == "company"]
        self.assertEqual(len(years), 2)
        for o in years:
            self.assertFalse(data.detect_years(o["question"]) and min(data.detect_years(o["question"])) < "2025")
        for o in comps:
            self.assertEqual(data.detect_companies(o["question"]), [])            # no corpus company left
            self.assertNotIn(o["source"], {y["source"] for y in years})            # different source questions
        jnj = [o for o in comps if "JnJ" in next(q["question"] for q in self.qs if q["financebench_id"] == o["source"])]
        if jnj:
            self.assertNotIn("JnJ", jnj[0]["question"])                           # every mention replaced

    def test_deterministic(self):
        self.assertEqual(data.make_unanswerable(self.qs, 2), data.make_unanswerable(self.qs, 2))


class Evidence(unittest.TestCase):
    def test_overlap_ignores_layout(self):
        page = "Total revenue\n$ 1,234\nNet income 567"
        self.assertEqual(data.evidence_overlap("total revenue $1,234", page), 1.0)
        self.assertLess(data.evidence_overlap("operating cash flow 999", page), 0.5)
        self.assertEqual(data.evidence_overlap("", page), 0.0)

    def test_doc_meta_from_info_and_from_name(self):
        info = {"3M_2018_10K": {"company": "3M", "doc_type": "10k", "doc_period": 2018}}
        self.assertEqual(data.doc_meta("3M_2018_10K", info), {"company": "3M", "doc_type": "10K", "period": "2018"})
        self.assertEqual(data.doc_meta("3M_2023Q2_10Q", info), {"company": "3M", "doc_type": "10Q", "period": "2023Q2"})


if __name__ == "__main__":
    unittest.main()
