import type { Track, Genre, RecommendationResult, ScoreBreakdown, SimilarityWeights } from './types'
import { computeGenreSimilarity, computeTextSimilarity, computeArtistBoost } from './similarity'

const DEFAULT_WEIGHTS: SimilarityWeights = {
  genre: 0.6,
  artist: 0.2,
  text: 0.2,
}

export class Recommender {
  private catalog: Track[]
  private weights: SimilarityWeights

  constructor(catalog: Track[], weights: SimilarityWeights = DEFAULT_WEIGHTS) {
    this.catalog = catalog
    this.weights = weights
  }

  setCatalog(catalog: Track[]) {
    this.catalog = catalog
  }

  getCatalog(): Track[] {
    return this.catalog
  }

  recommend(
    seed: Track,
    topK: number = 10,
    excludeSeed: boolean = true,
  ): RecommendationResult[] {
    const similarities: { track: Track; breakdown: ScoreBreakdown }[] = []

    for (const candidate of this.catalog) {
      if (excludeSeed && candidate.id === seed.id) continue

      const genreSim = computeGenreSimilarity(seed.genres, candidate.genres)
      const artistBoost = computeArtistBoost(seed, candidate)
      const textSim = computeTextSimilarity(seed, candidate)

      similarities.push({
        track: candidate,
        breakdown: { genreSimilarity: genreSim, artistBoost, textSimilarity: textSim },
      })
    }

    const scored: RecommendationResult[] = similarities.map(({ track, breakdown }) => {
      const score =
        this.weights.genre * breakdown.genreSimilarity +
        this.weights.artist * breakdown.artistBoost +
        this.weights.text * breakdown.textSimilarity
      return { track, score, breakdown }
    })

    scored.sort((a, b) => b.score - a.score)
    return scored.slice(0, topK)
  }

  recommendFromGenre(
    genres: Genre[],
    topK: number = 10,
    excludeIds: Set<string> = new Set(),
  ): RecommendationResult[] {
    const scored: RecommendationResult[] = this.catalog
      .filter(t => !excludeIds.has(t.id))
      .map(track => {
        const genreSim = computeGenreSimilarity(genres, track.genres)
        const breakdown: ScoreBreakdown = {
          genreSimilarity: genreSim,
          artistBoost: 0,
          textSimilarity: 0,
        }
        return { track, score: genreSim, breakdown }
      })

    scored.sort((a, b) => b.score - a.score)
    return scored.slice(0, topK)
  }
}
