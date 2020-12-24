def speechToText():
        import speech_recognition as sr

        sample_rate = 48000
        chunk_size = 2048

        mic_name='Microphone Array (Realtek(R) Au'
        mic_list=sr.Microphone.list_microphone_names()

        client= sr.Recognizer()

        for i in range(len(mic_list)):
                if(mic_list[i]==mic_name):
                        device_id=i

        with sr.Microphone(device_index=device_id,sample_rate=sample_rate,chunk_size=chunk_size) as source:
                client.adjust_for_ambient_noise(source)
                print("Say something: ")
                audio=client.listen(source)
                try:
                        text=client.recognize_google(audio)
                        return text
                except sr.UnknownValueError:
                        print("Could not make out audio")
                except sr.RequestError as e:
                        print("Failed to fetch")