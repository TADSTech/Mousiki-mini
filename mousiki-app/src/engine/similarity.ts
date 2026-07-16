import type { Genre, Track } from './types'

const GENRE_LIST: Genre[] = [
  'Rock', 'Pop', 'Jazz', 'Electronic', 'Hip-Hop',
  'Classical', 'R&B', 'Country', 'Metal', 'Reggae',
  'Indie', 'Blues', 'Folk', 'Latin', 'Soul',
  'Funk', 'Punk', 'Ambient', 'Pop Rock', 'Alternative',
]

export function genreVector(genres: Genre[]): number[] {
  return GENRE_LIST.map(g => genres.includes(g) ? 1 : 0)
}

export function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i]
    na += a[i] * a[i]
    nb += b[i] * b[i]
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb)
  return denom === 0 ? 0 : dot / denom
}

export function jaccardSimilarity(a: Genre[], b: Genre[]): number {
  const setA = new Set(a), setB = new Set(b)
  let intersection = 0
  for (const g of setA) if (setB.has(g)) intersection++
  const union = new Set([...setA, ...setB]).size
  return union === 0 ? 0 : intersection / union
}

export function textSimilarity(a: string, b: string): number {
  const tokensA = a.toLowerCase().split(/\s+/)
  const tokensB = b.toLowerCase().split(/\s+/)
  const setA = new Set(tokensA), setB = new Set(tokensB)
  let intersection = 0
  for (const t of setA) if (setB.has(t)) intersection++
  const union = new Set([...setA, ...setB]).size
  return union === 0 ? 0 : intersection / union
}

export function computeGenreSimilarity(a: Genre[], b: Genre[]): number {
  return jaccardSimilarity(a, b)
}

export function computeTextSimilarity(a: Track, b: Track): number {
  const textA = `${a.title} ${a.artist} ${(a.tags ?? []).join(' ')}`
  const textB = `${b.title} ${b.artist} ${(b.tags ?? []).join(' ')}`
  return textSimilarity(textA, textB)
}

export function computeArtistBoost(a: Track, b: Track): number {
  const aNorm = a.artist.toLowerCase().trim()
  const bNorm = b.artist.toLowerCase().trim()
  if (aNorm === bNorm) return 1.0
  if (aNorm.includes(bNorm) || bNorm.includes(aNorm)) return 0.5
  const aWords = new Set(aNorm.split(/\s+/))
  const bWords = bNorm.split(/\s+/)
  let shared = 0
  for (const w of bWords) if (aWords.has(w)) shared++
  return shared > 0 ? 0.25 : 0
}
