import { ReactNode } from 'react';
import {
  Box,
  Stack,
  HStack,
  Heading,
  Text,
  VStack,
  useColorModeValue,
  List,
  ListItem,
  ListIcon,
  Button,
  Container,
} from '@chakra-ui/react';
// import { FaCheckCircle } from 'react-icons/fa';

function PriceWrapper({ children }: { children: ReactNode }) {
  return (
    <Box
      mb={4}
      shadow="base"
      borderWidth="1px"
      alignSelf={{ base: 'center', lg: 'flex-start' }}
      borderColor={useColorModeValue('gray.200', 'gray.500')}
      borderRadius={'xl'}>
      {children}
    </Box>
  );
}

export default function ThreeTierPricing() {
  return (
    <Container maxW={'5xl'}>

      <Box py={12}>

        <VStack spacing={2} textAlign="center">
          <Heading 
          fontWeight={600}
          fontSize={{ base: '3xl', sm: '4xl', md: '6xl' }}
          lineHeight={'110%'}>
            Plans that fit your need
          </Heading>
          <Text fontSize="lg" color={'gray.500'}>
            Start with 14-day free trial. No credit card needed. Cancel at
            anytime.
          </Text>
        </VStack>

        <Stack
          direction={{ base: 'column', md: 'row' }}
          textAlign="center"
          justify="center"
          spacing={{ base: 4, lg: 10 }}
          py={10}>
          
          <PriceWrapper>
            <Box position="relative">
              <Box
                position="absolute"
                top="-16px"
                left="50%"
                style={{ transform: 'translate(-50%)' }}>
                <Text
                  textTransform="uppercase"
                  bg={useColorModeValue('green.300', 'green.700')}
                  px={3}
                  py={1}
                  color={useColorModeValue('gray.900', 'gray.300')}
                  fontSize="sm"
                  fontWeight="600"
                  rounded="xl">
                  Free Trial
                </Text>
              </Box>
              <Box py={4} px={12}>
                <Text fontWeight="500" fontSize="2xl">
                  Startup
                </Text>
                <HStack justifyContent="center">
                  <Text fontSize="3xl" fontWeight="600">
                    $
                  </Text>
                  <Text fontSize="5xl" fontWeight="900">
                    18
                  </Text>
                  <Text fontSize="3xl" color="gray.500">
                    /month
                  </Text>
                </HStack>
              </Box>
              <VStack
                bg={useColorModeValue('gray.50', 'gray.700')}
                py={4}
                borderBottomRadius={'xl'}>
                <List spacing={3} textAlign="start" px={12}>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ up to 3 users
                  </ListItem>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ connect chatbot to 10+ apps
                  </ListItem>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ advanced AI Agents powered by LLMs
                  </ListItem>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ Slackbot in unlimited channels
                  </ListItem>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ LLM selection (GPT-3.5, GPT-4, Claude, Llama2, etc.)
                  </ListItem>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ 2GB space storage
                  </ListItem>

                </List>
                <Box w="80%" pt={7}>
                  <Button w="full" colorScheme="orange" variant='outline'>
                    Start trial
                  </Button>
                </Box>
              </VStack>
            </Box>
          </PriceWrapper>

          <PriceWrapper>
            <Box position="relative">
              <Box py={4} px={12}>
                <Text fontWeight="500" fontSize="2xl">
                  Business
                </Text>
                <HStack justifyContent="center">
                  <Text fontSize="3xl" fontWeight="600">
                    $
                  </Text>
                  <Text fontSize="5xl" fontWeight="900">
                    56
                  </Text>
                  <Text fontSize="3xl" color="gray.500">
                    /month
                  </Text>
                </HStack>
              </Box>
              <VStack
                bg={useColorModeValue('gray.50', 'gray.700')}
                py={4}
                borderBottomRadius={'xl'}>
                <List spacing={3} textAlign="start" px={12}>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ up to 15 users
                  </ListItem>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ connect chatbot to 10+ apps
                  </ListItem>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ advanced AI Agents powered by LLMs
                  </ListItem>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ Slackbot in unlimited channels
                  </ListItem>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ LLM selection (GPT-3.5, GPT-4, Claude, Llama2, etc.)
                  </ListItem>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ 8GB space storage
                  </ListItem>
                </List>
                <Box w="80%" pt={7}>
                  <Button w="full" colorScheme="orange" >
                    Subscribe
                  </Button>
                </Box>
              </VStack>
            </Box>
          </PriceWrapper>
          
          <PriceWrapper>
            <Box position="relative">
              <Box py={4} px={12}>
                <Text fontWeight="500" fontSize="2xl">
                  Enterprise
                </Text>
                <HStack justifyContent="center">
                  <Text fontSize="3xl" fontWeight="600">
                    $
                  </Text>
                  <Text fontSize="5xl" fontWeight="900">
                    *
                  </Text>
                  <Text fontSize="3xl" color="gray.500">
                    month
                  </Text>
                </HStack>
              </Box>
              <VStack
                bg={useColorModeValue('gray.50', 'gray.700')}
                py={4}
                borderBottomRadius={'xl'}>
                <List spacing={3} textAlign="start" px={12}>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ up to 3 users
                  </ListItem>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ connect chatbot to 10+ apps
                  </ListItem>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ advanced AI Agents powered by LLMs
                  </ListItem>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ Slackbot in unlimited channels
                  </ListItem>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ Custom LLM integration (setup your custom model)
                  </ListItem>
                  <ListItem>
                    {/* <ListIcon as={FaCheckCircle} color="green.500" /> */}
                    ✔️ Unlimited storage space
                  </ListItem>
                </List>
                <Box w="80%" pt={7}>
                  <Button w="full" colorScheme="orange">
                    Contact Us
                  </Button>
                </Box>
              </VStack>
            </Box>
          </PriceWrapper>

        </Stack>

      </Box>
    </Container>

  );
}