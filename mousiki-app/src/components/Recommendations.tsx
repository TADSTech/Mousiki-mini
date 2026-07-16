import { useState } from 'react'
import { Sparkles, History, RefreshCcw } from 'lucide-react'
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
    <div className="glass-panel rounded-3xl p-6 md:p-8">
      <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4 mb-8">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-gradient-to-br from-amber-400 to-orange-500 rounded-xl text-white shadow-lg shadow-orange-500/20">
            <Sparkles className="w-6 h-6" />
          </div>
          <h2 className="text-2xl font-bold text-gray-50">Discovery Engine</h2>
        </div>
        
        {results.length > 0 && (
          <button 
            onClick={clearResults} 
            className="flex items-center gap-2 px-4 py-2 bg-gray-800/50 hover:bg-gray-700/50 rounded-lg text-sm font-medium text-gray-300 transition-colors border border-gray-700/50"
          >
            <RefreshCcw className="w-4 h-4" />
            Reset
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        {/* Seed Track Section */}
        <div className="bg-gray-900/30 p-5 rounded-2xl border border-gray-800/50">
          <div className="flex items-center gap-2 mb-4 text-gray-400 font-medium">
            <History className="w-4 h-4" />
            <span>Seed by Track</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {catalog.slice(0, 12).map(t => (
              <button
                key={t.id}
                onClick={() => handleTrackRecommend(t)}
                className={`px-3 py-1.5 rounded-xl text-sm font-medium transition-all duration-200 border
                  ${seedTrack?.id === t.id
                    ? 'bg-brand-600 border-brand-500 text-white shadow-lg shadow-brand-500/20 scale-105'
                    : 'bg-gray-800/50 border-gray-700/50 text-gray-300 hover:bg-gray-700 hover:border-gray-600'}`}
              >
                {t.title}
              </button>
            ))}
            {catalog.length === 0 && (
              <span className="text-sm text-gray-500 italic">Add tracks to use them as seeds...</span>
            )}
          </div>
        </div>

        {/* Seed Genre Section */}
        <div className="bg-gray-900/30 p-5 rounded-2xl border border-gray-800/50">
          <div className="flex items-center gap-2 mb-4 text-gray-400 font-medium">
            <Sparkles className="w-4 h-4" />
            <span>Seed by Genre</span>
          </div>
          <div className="flex flex-wrap gap-2 mb-4">
            {(['Rock', 'Pop', 'Jazz', 'Electronic', 'Hip-Hop', 'Classical', 'R&B', 'Country', 'Metal', 'Reggae'] as Genre[]).map(g => (
              <button
                key={g}
                type="button"
                onClick={() => toggleGenre(g)}
                className={`transition-all duration-200 ${
                  selectedGenres.includes(g) 
                    ? 'scale-105 shadow-md shadow-brand-500/10' 
                    : 'opacity-40 hover:opacity-80 grayscale hover:grayscale-0'
                }`}
              >
                <GenreBadge genre={g} />
              </button>
            ))}
          </div>
          <button
            onClick={handleGenreRecommend}
            disabled={selectedGenres.length === 0}
            className={`w-full py-2.5 rounded-xl font-semibold text-sm transition-all duration-300
              ${selectedGenres.length > 0
                ? 'bg-gradient-to-r from-gray-800 to-gray-700 border border-gray-600 text-white hover:border-gray-500 shadow-md cursor-pointer'
                : 'bg-gray-800/30 border border-gray-800/50 text-gray-600 cursor-not-allowed'}`}
          >
            Generate from Genres
          </button>
        </div>
      </div>

      {/* Results Section */}
      {results.length > 0 && (
        <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
          <div className="flex items-center gap-3 mb-6 pb-4 border-b border-gray-800">
            <div className="flex-1">
              <h3 className="text-xl font-bold text-gray-50">Generated Recommendations</h3>
              <p className="text-sm text-gray-400 mt-1">
                {genreMode
                  ? `Based on genres: ${selectedGenres.join(', ')}`
                  : `Inspired by: ${seedTrack?.title} — ${seedTrack?.artist}`}
              </p>
            </div>
            <div className="px-3 py-1 bg-brand-500/10 border border-brand-500/20 rounded-lg text-brand-400 text-sm font-medium">
              {results.length} results
            </div>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-5">
            {results.map((r, index) => (
              <div 
                key={r.track.id} 
                className="animate-in fade-in zoom-in-95 duration-500"
                style={{ animationDelay: `${index * 50}ms`, animationFillMode: 'both' }}
              >
                <TrackCard
                  track={r.track}
                  score={r.score}
                  breakdown={r.breakdown}
                  onClick={() => handleTrackRecommend(r.track)}
                />
              </div>
            ))}
          </div>
        </div>
      )}

      {results.length === 0 && (
        <div className="flex flex-col items-center justify-center py-12 text-center bg-gray-900/20 rounded-2xl border border-gray-800/50">
          <Sparkles className="w-12 h-12 text-gray-700 mb-4" />
          <p className="text-gray-400 font-medium">Select a track or genres above to discover new music</p>
          <p className="text-gray-500 text-sm mt-1">Our engine computes matches entirely in your browser.</p>
        </div>
      )}
    </div>
  )
}

export default Recommendations
