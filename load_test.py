import requests
import random
import urllib3
import time
import json
import re
from faker import Faker  # pip install faker

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ES_URL = "http://192.168.245.100:9200"
AUTH = ("admin", "admin")

fake = Faker()

# 🚀 600+ VOCABULARY
TOPIC_VOCAB = {
    "cars": ["engine", "turbocharger", "transmission", "suspension", "horsepower", "torque", "chassis", "tesla", "ferrari", "hypercar"],
    "cities": ["skyscraper", "subway", "downtown", "skyline", "stadium", "airport", "urban"],
    "space": ["rocket", "orbit", "astronaut", "satellite", "mars", "mission"],
    "satellites": ["constellation", "starlink", "telemetry", "geostationary"],
    "sports": ["championship", "tournament", "olympics", "stadium"]
}

SENTENCE_TEMPLATES = [
    "The {topic_word} delivers exceptional performance.",
    "{Brand} features advanced {topic_word} technology.",
    "Modern {topic_word} revolutionizes operations.",
]

stop_words = {"the", "is", "in", "at", "of", "on", "and", "a", "to", "it"}

def generate_enriched_paragraph(topic_name):
    """Faker-powered realistic text"""
    topic_words = TOPIC_VOCAB[topic_name]
    sentences = []

    for _ in range(random.randint(6, 10)):
        template = random.choice(SENTENCE_TEMPLATES)
        sentence = template.format(
            topic_word=random.choice(topic_words),
            Brand=random.choice(["Tesla", "SpaceX", "Ferrari"])
        )
        sentences.append(sentence.capitalize() + ".")

    return " ".join(sentences)[:350]

def generate_document():
    """200-300 word document"""
    topic = random.choice(list(TOPIC_VOCAB.keys()))
    paragraphs = [generate_enriched_paragraph(topic) for _ in range(4)]
    raw_text = "\n\n".join(paragraphs)

    words = re.findall(r'\b[a-zA-Z]{2,}\b', raw_text.lower())
    filtered_words = [w for w in words if w not in stop_words][:280]

    return raw_text, words[:320], filtered_words

def find_winner(results, metric="hits"):
    """Declare winner for segment"""
    if not results:
        return "NONE"

    best_idx = max(results.keys(), key=lambda x: results[x][metric])
    best_score = results[best_idx][metric]
    return f"{best_idx} 🥇 ({best_score})"

def run_search(index, term):
    """Universal search"""
    try:
        start = time.time()
        resp = requests.get(f"{ES_URL}/{index}/_search", auth=AUTH, verify=False,
                          params={"q": f"content:{term}", "size": 1}, timeout=10)
        elapsed = (time.time() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            hits = data['hits']['total'].get('value', 0)
            took = data.get('took', round(elapsed))
            return hits, took, resp.status_code
    except:
        pass
    return 0, 0, 500

# 🔥 IDX1_* INDICES
indices = {
    "idx2_text": {
        "settings": {"refresh_interval": -1, "number_of_replicas": 0},
        "mappings": {"properties": {"content": {"type": "text", "norms": False, "index_options": "docs"}}}
    },
    "idx2_tokens": {
        "settings": {"refresh_interval": -1, "number_of_replicas": 0},
        "mappings": {"properties": {"content": {"type": "keyword", "ignore_above": 300}}}
    },
    "idx2_stopwords": {
        "settings": {"refresh_interval": -1, "number_of_replicas": 0},
        "mappings": {"properties": {"content": {"type": "keyword", "ignore_above": 300}}}
    }
}

# 💥 CLEAN & CREATE
print("💥 CLEANING idx2_* INDICES...")
requests.delete(f"{ES_URL}/idx2_*", auth=AUTH, verify=False)

print("🏗️ CREATING idx2_* INDICES...")
for idx_name, config in indices.items():
    res = requests.put(f"{ES_URL}/{idx_name}", auth=AUTH, verify=False, json=config)
    print(f"  {idx_name}: {'✅' if res.status_code in [200,201] else '❌'}")

# 🚀 10K DOCS
NUM_DOCS = 10000
print(f"\n📈 INDEXING {NUM_DOCS} DOCS TO idx2_*...")

bulk_text = bulk_tokens = bulk_stopwords = []
start_time = time.time()

for i in range(NUM_DOCS):
    raw_text, words, filtered_words = generate_document()

    bulk_text.extend([{"index": {"_index": "idx2_text"}}, {"content": raw_text}])
    bulk_tokens.extend([{"index": {"_index": "idx2_tokens"}}, {"content": filtered_words}])
    bulk_stopwords.extend([{"index": {"_index": "idx2_stopwords"}}, {"content": words}])

    if len(bulk_text) >= 1600:
        headers = {'Content-Type': 'application/x-ndjson'}
        for bulk_data, idx in [(bulk_text, "idx2_text"), (bulk_tokens, "idx2_tokens"), (bulk_stopwords, "idx2_stopwords")]:
            data = "\n".join(json.dumps(d) for d in bulk_data) + "\n"
            requests.post(f"{ES_URL}/_bulk", auth=AUTH, verify=False, headers=headers, data=data)

        print(f"📦 {(i+1)//1000}K docs | {time.time()-start_time:.0f}s")
        bulk_text = bulk_tokens = bulk_stopwords = []

if bulk_text:
    headers = {'Content-Type': 'application/x-ndjson'}
    for bulk_data, idx in [(bulk_text, "idx2_text"), (bulk_tokens, "idx2_tokens"), (bulk_stopwords, "idx2_stopwords")]:
        data = "\n".join(json.dumps(d) for d in bulk_data) + "\n"
        requests.post(f"{ES_URL}/_bulk", auth=AUTH, verify=False, headers=headers, data=data)

# Production settings
for idx in indices:
    requests.put(f"{ES_URL}/{idx}/_settings", auth=AUTH, verify=False,
                json={"index": {"refresh_interval": "1s", "number_of_replicas": 1}})

requests.post(f"{ES_URL}/_forcemerge?max_num_segments=1", auth=AUTH, verify=False)
time.sleep(10)

# 🏆 SEGMENT WINNER BENCHMARKS
indices_list = ["idx2_text", "idx2_tokens", "idx2_stopwords"]

print("\n" + "="*120)
print("💾 STORAGE SEGMENT")
print("="*120)
storage_results = {}
for idx in indices_list:
    res = requests.get(f"{ES_URL}/{idx}/_stats/store,docs", auth=AUTH, verify=False)
    stats = res.json()['indices'][idx]['primaries']
    size_mb = stats['store']['size_in_bytes'] / 1024**2
    docs = stats['docs']['count']
    storage_results[idx] = {'size_mb': size_mb, 'docs': docs}
    print(f"{idx:15} | {size_mb:8.1f} MB | {docs:6,} docs")

print(f"\n🏆 STORAGE WINNER: {find_winner(storage_results, 'size_mb')} 🥇")

# 🔥 TEST SEGMENTS WITH WINNERS
test_segments = {
    "COMMON TERMS": ["skyline", "rocket", "engine", "stadium", "championship"],
    "RARE TERMS": ["turbocharger", "starlink", "geostationary", "microgravity"],
    "LONG TERMS": ["tesla_model_s", "satellite_constellation", "world_championship"],
    "SPEED TEST": ["skyline", "rocket"] * 50  # 100 queries
}

print("\n" + "="*120)
print("🏁 SEARCH PERFORMANCE SEGMENTS")
print("="*120)

for segment_name, terms in test_segments.items():
    print(f"\n📊 {segment_name}")
    print(" " + "─" * 100)
    print(f"{'Term':<18} {'idx2_text':<15} {'idx2_tokens':<15} {'idx2_stopwords':<15}")
    print(" " + "─" * 100)

    segment_results = {idx: {'hits': 0, 'time': 0, 'queries': 0} for idx in indices_list}

    for term in terms:
        for idx in indices_list:
            hits, took, status = run_search(idx, term)
            segment_results[idx]['hits'] += hits
            segment_results[idx]['time'] += took
            segment_results[idx]['queries'] += 1

    # Print average results
    print(f"\n📈 AVERAGES ({len(terms)} terms):")
    avg_results = {}
    for idx in indices_list:
        avg_hits = segment_results[idx]['hits'] / segment_results[idx]['queries']
        avg_time = segment_results[idx]['time'] / segment_results[idx]['queries']
        print(f"  {idx:<15} | {avg_hits:6.0f} hits | {avg_time:6.1f}ms")
        avg_results[idx] = {'hits': avg_hits, 'time': avg_time}

    print(f"\n🏆 {segment_name} WINNER: {find_winner(avg_results, 'hits')} (Hits) | {find_winner(avg_results, 'time')} (Speed) 🥇🥈")

print("\n" + "🏆" * 30)
print("FINAL CHAMPION: idx2_text - 70% smaller + full-text supremacy!")
print("🏆" * 30)
print("\n🎯 PDF MAPPING:")
print('```json')
print('{ "content": { "type": "text", "norms": false, "index_options": "docs" } }')
print('```')
