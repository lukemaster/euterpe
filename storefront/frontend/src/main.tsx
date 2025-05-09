/*
 * Copyright (C) 2025 Rafael Luque Tejada
 * Author: Rafael Luque Tejada <lukemaster.master@gmail.com>
 *
 * This file is part of Generación de Música Personalizada a través de Modelos Generativos Adversariales.
 *
 * Euterpe as a part of the project Generación de Música Personalizada a través de Modelos Generativos Adversariales is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Generación de Música Personalizada a través de Modelos Generativos Adversariales is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


import React from 'react'
import ReactDOM from 'react-dom/client'
// import EuterpeApp from './components/EuterpeApp'
import EuterpeApp from './components/EuterpeApp.functional'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <EuterpeApp />
  </React.StrictMode>
)
