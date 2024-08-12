<template>
  <nav class="bg-gray-800 p-4 shadow-md">
    <div class="container mx-auto flex items-center justify-between">
      <!-- Logo/Brand Name -->
      <div class="text-white text-2xl font-bold tracking-wide">MyApp</div>

      <!-- Desktop Menu -->
      <div class=" md:flex md:items-center md:space-x-6">
        <a href="/"
          class="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition duration-150">Home</a>
        <a href="/upload"
          class="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition duration-150">Upload</a>
        <a href="/recommendation"
          class="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition duration-150">Recommendation</a>
        <a href="/history"
          class="text-gray-300 hover:text-white px-3 py-2 rounded-md text-sm font-medium transition duration-150">History</a>
      </div>

      <!-- Mobile Menu Button -->
      <Menu as="div" class="relative md:hidden">
        <MenuButton
          class="text-gray-400 hover:text-white focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-white">
          <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"></path>
          </svg>
        </MenuButton>
        <TransitionRoot enter="transition ease-out duration-100" enter-from="transform opacity-0 scale-95"
          enter-to="transform opacity-100 scale-100" leave="transition ease-in duration-75"
          leave-from="transform opacity-100 scale-100" leave-to="transform opacity-0 scale-95">
          <MenuItems
            class="absolute right-0 mt-2 w-48 origin-top-right bg-white divide-y divide-gray-100 rounded-md shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
            <div class="p-1">
              <MenuItem>
              {({ active }) => (
              <a href="/" class="block px-4 py-2 text-sm" :class="{
                'bg-gray-100 text-gray-900': active,
                'text-gray-700': !active,
              }">
                Home
              </a>
              )}
              </MenuItem>
              <MenuItem>
              {({ active }) => (
              <a href="/upload" class="block px-4 py-2 text-sm" :class="{
                'bg-gray-100 text-gray-900': active,
                'text-gray-700': !active,
              }">
                Upload
              </a>
              )}
              </MenuItem>
              <MenuItem>
              {({ active }) => (
              <a href="/recommendation" class="block px-4 py-2 text-sm" :class="{
                'bg-gray-100 text-gray-900': active,
                'text-gray-700': !active,
              }">
                Recommendation
              </a>
              )}
              </MenuItem>
              <MenuItem>
              {({ active }) => (
              <a href="/history" class="block px-4 py-2 text-sm" :class="{
                'bg-gray-100 text-gray-900': active,
                'text-gray-700': !active,
              }">
                History
              </a>
              )}
              </MenuItem>
            </div>
          </MenuItems>
        </TransitionRoot>
      </Menu>
    </div>
  </nav>
</template>

<script setup>
import { Menu, MenuButton, MenuItems, MenuItem } from "@headlessui/vue";
import { TransitionRoot } from "@headlessui/vue";
</script>

<style scoped>
/* Add any additional styles here if needed */
</style>
