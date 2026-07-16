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
    <div className="min-h-screen bg-gray-950 text-gray-50 relative overflow-hidden selection:bg-brand-500/30">
      {/* Dynamic Background Glows */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-brand-600/20 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-accent-600/20 rounded-full blur-[120px] pointer-events-none" />

      <div className="max-w-7xl mx-auto px-4 py-8 relative z-10">
        <Header trackCount={catalog.length} />

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8 mt-8">
          <div className="lg:col-span-3 space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
            <MusicLibrary
              tracks={catalog}
              selectedTrack={selectedTrack}
              onSelectTrack={handleSelectTrack}
              onRemoveTrack={handleRemoveTrack}
            />
          </div>

          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-8 duration-700 delay-150">
            <AddMusicForm onAdd={handleAddTrack} />
          </div>
        </div>

        <div className="mt-8 animate-in fade-in slide-in-from-bottom-12 duration-700 delay-300">
          <Recommendations recommender={recommender} catalog={catalog} />
        </div>

        <footer className="mt-16 text-center text-sm text-gray-500 border-t border-gray-800/50 pt-8 pb-4">
          <p className="flex items-center justify-center gap-2">
            <span className="font-medium text-gray-400">Mousiki</span> 
            <span>&mdash;</span>
            Built with React, TypeScript & Tailwind CSS.
          </p>
          <p className="mt-2 text-xs opacity-60">All computation runs client-side. No backend.</p>
        </footer>
      </div>
    </div>
  )
}

export default App
