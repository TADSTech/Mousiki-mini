export interface Track {
  id: string
  title: string
  artist: string
  genres: Genre[]
  tags?: string[]
}

export type Genre =
  | 'Rock' | 'Pop' | 'Jazz' | 'Electronic' | 'Hip-Hop'
  | 'Classical' | 'R&B' | 'Country' | 'Metal' | 'Reggae'
  | 'Indie' | 'Blues' | 'Folk' | 'Latin' | 'Soul'
  | 'Funk' | 'Punk' | 'Ambient' | 'Pop Rock' | 'Alternative'

export interface ScoreBreakdown {
  genreSimilarity: number
  artistBoost: number
  textSimilarity: number
}

export interface RecommendationResult {
  track: Track
  score: number
  breakdown: ScoreBreakdown
}

export interface SimilarityWeights {
  genre: number
  artist: number
  text: number
}
