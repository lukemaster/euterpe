// EuterpeApp.tsx - real version connected with Flask API
import React, { useRef, useState, useEffect } from 'react'

interface GeneratedTruck {
  id: number
  name: string
  genre: string
  url: string
}

const Button = (props: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
  <button className='text-lg px-6 py-3 rounded bg-green-700 text-green-100 hover:bg-green-600 disabled:opacity-50' {...props} />
)

const availableGenres = ['jazz', 'rock', 'clásica', 'electrónica', 'lo-fi', 'ambient']

export default function EuterpeApp(): JSX.Element {
  const [genre, setGenre] = useState<string>('')
  const [generatedTruck, setGeneratedTruck] = useState<GeneratedTruck | null>(null)
  const [rating, setRate] = useState<number | null>(null)
  const [state, setState] = useState<'idle' | 'generating' | 'generated' | 'rated'>('idle')
  const helpRef = useRef<HTMLDivElement | null>(null)
  const [help, setHelp] = useState<string>('')

  useEffect(() => {
    fetch('/api/help')
      .then((res) => res.json())
      .then((data) => setHelp(data.texto))
  }, [])


  const generateTruck = async (): Promise<void> => {
    setState('generating')
    await new Promise((res) => setTimeout(res, 1500))
    const name = `${genre}_${Date.now()}`
    const data: GeneratedTruck = {
      id: Math.floor(Math.random() * 1000),
      name,
      genre,
      url: '/static/generated/fake.mp3'
    }
    setGeneratedTruck(data)
    setState('generated')
  }

  const sendRating = async (): Promise<void> => {
    setState('rated')
    setTimeout(() => {
      setState('idle')
      setGenre('')
      setGeneratedTruck(null)
      setRate(null)
    }, 5000)
  }
  const scrollAyuda = (): void => {
    setTimeout(() => helpRef.current?.scrollIntoView({ behavior: 'smooth' }), 100)
  }

  return (
    <main className='w-full min-h-screen bg-black text-green-400 flex flex-col items-center px-8 py-12 space-y-10'>
      <img src='images/temple.png' alt='Euterpe' className='hidden md:block fixed top-0 left-0 w-65 rounded-lg shadow-lg z-10' />
      <img src='images/eu2.png' alt='Euterpe' className='hidden md:block fixed top-6 right-60 w-80 rounded-lg shadow-lg z-10' />
      
      <div className='bg-black max-w-xl w-full space-y-10'>
        <div className='flex flex-col items-start w-full space-y-2 pt-0'>
          <h1 className='text-4xl font-bold text-green-500'>Ευτέρπη</h1>
          <h2 className='text-4xl font-bold text-green-500'>(Euterpe)</h2>
        </div>

        <div>
          <p className='text-xl font-bold text-green-500'>Euterpe, musa de la música e inventora de las matemáticas, alza su canto artificial en este templo digital.</p>
          <p className='text-xl font-bold text-green-500'>Desde los números y el ritmo, inspira la IA para ti en composiciones únicas, tejiendo armonía con lógica.</p>
          <p className='text-xl font-bold text-green-500'>Tú solo escuchas. Ella compone.</p>
        </div>

        <section className='space-y-6'>
          <div className='text-xl space-y-4'>
            <p className='font-semibold'>Selecciona un género musical:</p>
            <div className='space-y-3'>
              {availableGenres.map((g) => (
                <label key={g} className='block text-lg'>
                  <input
                    type='radio'
                    name='genero'
                    value={g}
                    checked={genre === g}
                    onChange={(e) => setGenre(e.target.value)}
                    disabled={state !== 'idle'}
                    className='mr-3 scale-150'
                  />
                  {g}
                </label>
              ))}
            </div>
            <div className='text-center'>
              <Button onClick={generateTruck} disabled={!genre || state !== 'idle'}>
                {state === 'generating' ? 'Generando...' : 'Generar'}
              </Button>
            </div>
          </div>

          {generatedTruck && state !== 'idle' && (
            <div className='space-y-4'>
              <p className='text-lg'>Pieza generada: <strong>{generatedTruck.name}</strong> ({generatedTruck.genre})</p>
              <audio controls src={generatedTruck.url} className='w-full' />

              {state === 'generated' && (
                <>
                  <p className='text-lg font-semibold'>Valora la pieza (1 a 10):</p>
                  <div className='flex flex-wrap gap-4 text-xl'>
                    {Array.from({ length: 10 }, (_, i) => i + 1).map((v) => (
                      <label key={v}>
                        <input
                          type='radio'
                          name='rating'
                          value={v}
                          checked={rating === v}
                          onChange={() => setRate(v)}
                          className='mr-2 scale-125'
                        />
                        {v}
                      </label>
                    ))}
                  </div>
                  <Button onClick={sendRating} disabled={rating === null}>Mandar valoración</Button>
                </>
              )}

              {state === 'rated' && <p className='text-green-500 text-lg'>Gracias por tu valoración. Puedes generar otra pieza en unos segundos...</p>}
            </div>
          )}

          <div className='pt-10 text-left space-y-4'>
            <div className='text-center'>
              <Button onClick={scrollAyuda}>Mostrar Ayuda</Button>
            </div>
          </div>

          <div ref={helpRef} className='space-y-4 text-lg pt-80 whitespace-pre-line'>
            {help.split('\n').map((line, idx) => (
              <p key={idx}>{line}</p>
            ))}
          </div>
        </section>
      </div>
    </main>
  )
}