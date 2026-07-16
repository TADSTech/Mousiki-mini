import { useState, useMemo } from 'react'
import type { Track } from './engine/types'
import { DEFAULT_CATALOG } from './engine/catalog'
import { Recommender } from './engine/recommender'
import Header from './components/Header'
import MusicLibrary from './components/MusicLibrary'
import AddMusicForm from './components/AddMusicForm'
import Recommendations from './components/Recommendations'

const App = () => {
  const [catalog, setCatalog] = useState<Track[]>(DEFAULT_CATALOG)
  const [selectedTrack, setSelectedTrack] = useState<Track | null>(null)

  const recommender = useMemo(() => new Recommender(catalog), [catalog])

  const handleAddTrack = (track: Track) => {
    const updated = [...catalog, track]
    setCatalog(updated)
    setSelectedTrack(track)
  }

  const handleRemoveTrack = (id: string) => {
    setCatalog(prev => prev.filter(t => t.id !== id))
    if (selectedTrack?.id === id) setSelectedTrack(null)
  }

  const handleSelectTrack = (track: Track) => {
    setSelectedTrack(track)
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <div className="max-w-7xl mx-auto px-4 py-8">
        <Header trackCount={catalog.length} />

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          <div className="lg:col-span-3 space-y-6">
            <MusicLibrary
              tracks={catalog}
              selectedTrack={selectedTrack}
              onSelectTrack={handleSelectTrack}
              onRemoveTrack={handleRemoveTrack}
            />
          </div>

          <div className="space-y-6">
            <AddMusicForm onAdd={handleAddTrack} />
          </div>
        </div>

        <div className="mt-6">
          <Recommendations recommender={recommender} catalog={catalog} />
        </div>

        <footer className="mt-12 text-center text-xs text-gray-600 border-t border-gray-800 pt-6">
          Mousiki — built with React + TypeScript + Tailwind CSS. All computation runs client-side.
        </footer>
      </div>
    </div>
  )
}

export default App
