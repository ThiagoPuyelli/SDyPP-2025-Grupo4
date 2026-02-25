# Build stage
FROM node:20-alpine AS build

WORKDIR /app
COPY package*.json ./
RUN npm install

COPY . .
RUN npm run build

# Runtime stage
FROM nginx:alpine

# Borramos config default
RUN rm /etc/nginx/conf.d/default.conf

# Config custom
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Archivos estáticos
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080
