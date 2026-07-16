import { useState } from 'react'
import type { Track, Genre, RecommendationResult } from '../engine/types'
import { Recommender } from '../engine/recommender'
import TrackCard from './TrackCard'
import GenreBadge from './GenreBadge'

interface RecommendationsProps {
  recommender: Recommender
  catalog: Track[]
}

const Recommendations = ({ recommender, catalog }: RecommendationsProps) => {
  const [seedTrack, setSeedTrack] = useState<Track | null>(null)
  const [results, setResults] = useState<RecommendationResult[]>([])
  const [genreMode, setGenreMode] = useState(false)
  const [selectedGenres, setSelectedGenres] = useState<Genre[]>([])

  const handleTrackRecommend = (track: Track) => {
    setSeedTrack(track)
    setGenreMode(false)
    const recs = recommender.recommend(track, 12)
    setResults(recs)
  }

  const handleGenreRecommend = () => {
    if (selectedGenres.length === 0) return
    setSeedTrack(null)
    setGenreMode(true)
    const recs = recommender.recommendFromGenre(selectedGenres, 12)
    setResults(recs)
  }

  const toggleGenre = (g: Genre) => {
    setSelectedGenres(prev =>
      prev.includes(g) ? prev.filter(x => x !== g) : [...prev, g]
    )
  }

  const clearResults = () => {
    setResults([])
    setSeedTrack(null)
    setGenreMode(false)
  }

  return (
    <div className="bg-gray-800/30 border border-gray-700 rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white">Recommendations</h2>
        {results.length > 0 && (
          <button onClick={clearResults} className="text-sm text-gray-400 hover:text-white transition-colors">
            Clear
          </button>
        )}
      </div>

      <div className="space-y-3 mb-4">
        <p className="text-sm text-gray-400">Click any track to get recommendations based on it:</p>

        <div className="flex flex-wrap gap-1.5">
          {catalog.slice(0, 12).map(t => (
            <button
              key={t.id}
              onClick={() => handleTrackRecommend(t)}
              className={`px-2 py-1 rounded-lg text-xs font-medium transition-all
                ${seedTrack?.id === t.id
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'}`}
            >
              {t.title}
            </button>
          ))}
        </div>
      </div>

      <div className="border-t border-gray-700 pt-4 mb-4">
        <p className="text-sm text-gray-400 mb-2">Or recommend by genre:</p>
        <div className="flex flex-wrap gap-1.5 mb-3">
          {(['Rock', 'Pop', 'Jazz', 'Electronic', 'Hip-Hop', 'Classical', 'R&B', 'Country', 'Metal', 'Reggae'] as Genre[]).map(g => (
            <button
              key={g}
              type="button"
              onClick={() => toggleGenre(g)}
              className={selectedGenres.includes(g) ? '' : 'opacity-50 hover:opacity-80'}
            >
              <GenreBadge genre={g} />
            </button>
          ))}
        </div>
        <button
          onClick={handleGenreRecommend}
          disabled={selectedGenres.length === 0}
          className={`text-sm py-1.5 px-3 rounded-lg font-medium transition-all
            ${selectedGenres.length > 0
              ? 'bg-purple-600 hover:bg-purple-500 text-white cursor-pointer'
              : 'bg-gray-700 text-gray-500 cursor-not-allowed'}`}
        >
          Get Recommendations
        </button>
      </div>

      {results.length > 0 && (
        <div className="border-t border-gray-700 pt-4">
          <p className="text-sm text-gray-400 mb-3">
            {genreMode
              ? `Recommended for genres: ${selectedGenres.join(', ')}`
              : `Recommended based on: ${seedTrack?.title} — ${seedTrack?.artist}`}
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-3">
            {results.map(r => (
              <TrackCard
                key={r.track.id}
                track={r.track}
                score={r.score}
                breakdown={r.breakdown}
                onClick={() => handleTrackRecommend(r.track)}
              />
            ))}
          </div>
        </div>
      )}

      {results.length === 0 && (
        <p className="text-gray-500 text-sm text-center py-6">
          Select a track or genre above to see recommendations
        </p>
      )}
    </div>
  )
}

export default Recommendations
